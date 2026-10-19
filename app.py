import hashlib
import json
import time
import streamlit as st
import libsql_client
from google import genai
from google.genai import types

# --- Mobile-First Page Configuration ---
st.set_page_config(
    page_title="Murli Search AI",
    page_icon="🕊️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🕊️ Murli Search Assistant")
st.caption("Deep Semantic & Section-Aware Search across 8,200+ Sakar & Avyakt Murlis.")

# --- Retrieve & Validate Secrets ---
gemini_key = st.secrets.get("GEMINI_API_KEY")
turso_url = st.secrets.get("TURSO_DATABASE_URL")
turso_token = st.secrets.get("TURSO_AUTH_TOKEN")

if not (gemini_key and turso_url and turso_token):
    st.error("Missing configuration. Please check GEMINI_API_KEY, TURSO_DATABASE_URL, and TURSO_AUTH_TOKEN in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=gemini_key.strip())

# Explicitly use the model requested by the API error logs
MODEL_NAME = "gemini-3.8-flash"

SECTION_MARKERS = {
    "vardaan": ["वरदान:", "वरदान :-", "वरदान-", "वरदान :", "वरदान"],
    "slogan": ["स्लोगन:", "स्लोगन :-", "स्लोगन-", "स्लोगन :", "स्लोगन"],
    "dharna": ["धारणा के लिए मुख्य सार", "धारणा के लिए", "धारणा:-", "धारणा :"],
    "saar": ["मीठे बच्चे", "प्रातःमुरली", "मुख्य सार"]
}


def call_gemini_resilient(contents, config=None):
    """Calls Gemini API using gemini-3.8-flash with fallback options."""
    models_to_try = [MODEL_NAME, "gemini-2.5-flash", "gemini-flash"]
    last_error = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except Exception as e:
                last_error = e
                err_str = str(e).upper()
                if any(k in err_str for k in ["503", "UNAVAILABLE", "OVERLOADED", "HIGH DEMAND", "429", "404", "NOT_FOUND"]):
                    time.sleep(1.2 * (attempt + 1))
                    continue
                else:
                    raise e
    raise last_error


def get_turso_client():
    """Connects to Turso via stateless HTTPS."""
    http_url = turso_url.strip().replace("libsql://", "https://")
    return libsql_client.create_client_sync(url=http_url, auth_token=turso_token.strip())


def extract_search_intent(user_query: str, ui_section_override: str = "All Sections") -> dict:
    """Extracts core concepts, semantic Yuktis, and section filters."""
    prompt = f"""
You are a senior Brahma Kumaris Murli scholar. Analyze this user query: "{user_query}"

Return a valid JSON object with these exact keys:
1. "collection": "avyakt" (if user asks for Avyakt Murli / BapDada), "daily" (if Sakar / Daily Murli), or null.
2. "section_focus": "vardaan" (if asking for a Vardaan / blessing), "slogan" (if asking for a Slogan), "dharna" (if asking for Dharna points), "saar" (if asking for Mukhya Saar / essence), or null.
3. "date_filter": string if a specific year or date is mentioned, else null.
4. "core_concepts": list of 1 to 3 essential Hindi words/roots in Devanagari (e.g., "वैराग्य", "दुनिया", "निशानी").
5. "synonyms": list of 2 to 4 alternative Hindi spellings or closely related Murli terms in Devanagari.
6. "semantic_yuktis": list of 2 to 3 deeper BK Murli metaphors or Yuktis Baba uses for this topic in Devanagari.

Output ONLY valid JSON.
"""
    try:
        response = call_gemini_resilient(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        data = json.loads(response.text.strip())
    except Exception:
        stop_words = {"किस", "अव्यक्त", "साकार", "मुरली", "में", "बाबा", "ने", "कहा", "है", "कि", "से", "ही", "क्या", "कैसे"}
        words = [w.strip() for w in user_query.split() if w.strip() not in stop_words and len(w.strip()) > 2]
        data = {
            "collection": "avyakt" if ("अव्यक्त" in user_query or "avyakt" in user_query.lower()) else None,
            "section_focus": None,
            "date_filter": None,
            "core_concepts": words[:3] if words else [user_query.strip()],
            "synonyms": [],
            "semantic_yuktis": []
        }

    ui_map = {
        "वरदान (Vardaan)": "vardaan",
        "स्लोगन (Slogan)": "slogan",
        "धारणा (Dharna)": "dharna",
        "मुख्य सार (Essence)": "saar"
    }
    if ui_section_override in ui_map:
        data["section_focus"] = ui_map[ui_section_override]

    if not data.get("core_concepts"):
        data["core_concepts"] = [user_query.strip()]

    return data


def parse_murli_sections(full_text: str) -> dict:
    """Isolates Mukhya Saar, Dharna, Vardaan, and Slogan from the Murli text."""
    if not full_text:
        return {"saar": "", "dharna": "", "vardaan": "", "slogan": ""}

    sections = {
        "saar": full_text[:1000].strip(),
        "dharna": "",
        "vardaan": "",
        "slogan": ""
    }

    tail_start = max(0, int(len(full_text) * 0.55))
    tail_text = full_text[tail_start:]

    for marker in SECTION_MARKERS["dharna"]:
        idx = tail_text.find(marker)
        if idx != -1:
            sections["dharna"] = tail_text[idx:idx + 1200].strip()
            break

    for marker in SECTION_MARKERS["vardaan"]:
        idx = tail_text.rfind(marker)
        if idx != -1:
            sections["vardaan"] = tail_text[idx:idx + 900].strip()
            break

    for marker in SECTION_MARKERS["slogan"]:
        idx = tail_text.rfind(marker)
        if idx != -1:
            sections["slogan"] = tail_text[idx:idx + 400].strip()
            break

    return sections


def score_and_extract_passages(full_text: str, intent: dict, window_chars: int = 950, max_windows: int = 6) -> tuple[str, float]:
    """Scores Murlis and extracts up to 6 high-density passages + footers."""
    if not full_text:
        return "", 0.0

    core = [t for t in intent.get("core_concepts", []) if t]
    syns = [t for t in intent.get("synonyms", []) if t]
    yuktis = [t for t in intent.get("semantic_yuktis", []) if t]
    section_focus = intent.get("section_focus")

    all_search_terms = list(dict.fromkeys(core + syns + yuktis))
    sections = parse_murli_sections(full_text)

    score = 0.0
    for c in core:
        count = full_text.count(c)
        if count > 0:
            score += 5.0 + min(count * 1.2, 6.0)
    for s in syns:
        count = full_text.count(s)
        if count > 0:
            score += 3.0 + min(count * 0.8, 4.0)
    for y in yuktis:
        if y in full_text:
            score += 2.0

    focused_section_text = ""
    if section_focus and sections.get(section_focus):
        focused_section_text = sections[section_focus]
        score += 4.0
        for term in all_search_terms:
            if term in focused_section_text:
                score += 8.0

    step = 750
    chunk_len = window_chars * 2
    scored_chunks = []

    for start in range(0, max(1, len(full_text) - 200), step):
        end = min(len(full_text), start + chunk_len)
        chunk = full_text[start:end]
        c_score = 0.0
        unique_hits = 0
        for c in core:
            if c in chunk:
                c_score += 5.0
                unique_hits += 1
        for s in syns:
            if s in chunk:
                c_score += 3.0
                unique_hits += 1
        for y in yuktis:
            if y in chunk:
                c_score += 1.5
        if unique_hits >= 2:
            c_score += unique_hits * 4.0
        if c_score > 0:
            scored_chunks.append((c_score, start, end, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    selected_passages = []
    used_ranges = []
    for c_score, s, e, chunk in scored_chunks:
        if any(not (e < us or s > ue) for us, ue in used_ranges):
            continue
        used_ranges.append((s, e))
        prefix = "..." if s > 0 else ""
        suffix = "..." if e < len(full_text) else ""
        selected_passages.append(f"{prefix}{chunk.strip()}{suffix}")
        if len(selected_passages) >= max_windows:
            break

    if scored_chunks:
        score += scored_chunks[0][0]

    blocks = []
    if focused_section_text:
        blocks.append(f"🎯 [TARGETED SECTION — {section_focus.upper()}]:\n{focused_section_text}")

    if selected_passages:
        blocks.append("📜 [MATCHING MURLI PARAGRAPHS]:\n" + "\n\n---\n\n".join(selected_passages))
    else:
        blocks.append(full_text[:1800])

    footer_parts = []
    if sections["vardaan"] and section_focus != "vardaan":
        footer_parts.append(f"✨ [वरदान / Vardaan]: {sections['vardaan'][:450]}")
    if sections["slogan"] and section_focus != "slogan":
        footer_parts.append(f"🔔 [स्लोगन / Slogan]: {sections['slogan'][:250]}")
    if footer_parts:
        blocks.append("\n".join(footer_parts))

    return "\n\n".join(blocks), score


def search_turso(intent: dict, final_limit: int = 20) -> tuple[list[dict], str]:
    """Retrieves candidate Murlis and returns top 20 re-ranked results."""
    core = [t.replace('"', '').replace("'", "").strip() for t in intent.get("core_concepts", []) if t.strip()]
    syns = [t.replace('"', '').replace("'", "").strip() for t in intent.get("synonyms", []) if t.strip()]
    yuktis = [t.replace('"', '').replace("'", "").strip() for t in intent.get("semantic_yuktis", []) if t.strip()]

    section_focus = intent.get("section_focus")
    section_hindi_map = {
        "vardaan": "वरदान",
        "slogan": "स्लोगन",
        "dharna": "धारणा",
        "saar": "मीठे बच्चे"
    }
    section_term = section_hindi_map.get(section_focus)

    all_terms = list(dict.fromkeys(core + syns + yuktis))
    if not all_terms:
        return [], "none"

    fts_tiers = []
    if section_term and core:
        core_or_syn = " OR ".join([f'"{t}"' for t in (core + syns)])
        fts_tiers.append((f"Section-Targeted ({section_term})", f'"{section_term}" AND ({core_or_syn})'))
    if len(core) >= 2:
        fts_tiers.append(("Strict Concept Match", " AND ".join([f'"{c}"' for c in core])))
    if len(core) >= 1 and len(all_terms) >= 2:
        others = " OR ".join([f'"{t}"' for t in all_terms[1:]])
        fts_tiers.append(("Contextual + Synonym Match", f'"{core[0]}" AND ({others})'))
    fts_tiers.append(("Semantic & Yukti Expansion", " OR ".join([f'"{t}"' for t in all_terms])))

    coll_target = intent.get("collection")
    date_target = intent.get("date_filter")
    candidate_pool_size = 50

    def fetch_candidates(enforce_collection: bool) -> tuple[list, str]:
        where_clauses = ["murlis_fts MATCH ?"]
        extra_params = []
        if enforce_collection and coll_target in ["avyakt", "daily"]:
            where_clauses.append("LOWER(m.collection) LIKE ?")
            extra_params.append(f"%{coll_target.lower()}%")
        if date_target:
            where_clauses.append("m.murli_date LIKE ?")
            extra_params.append(f"%{date_target}%")

        sql = f"""
            SELECT 
                m.murli_date,
                m.collection,
                snippet(murlis_fts, 0, ' **', '** ', '...', 45) AS match_snippet,
                m.content
            FROM murlis_fts
            JOIN murlis m ON m.id = murlis_fts.rowid
            WHERE {" AND ".join(where_clauses)}
            ORDER BY rank
            LIMIT ?;
        """

        candidates = []
        seen_keys = set()
        primary_strategy = ""
        with get_turso_client() as db:
            for label, fts_q in fts_tiers:
                try:
                    params = [fts_q] + extra_params + [candidate_pool_size]
                    res = db.execute(sql, params)
                    if res.rows:
                        if not primary_strategy:
                            primary_strategy = label
                        for r in res.rows:
                            key = (str(r[0]), str(r[1]))
                            if key not in seen_keys:
                                seen_keys.add(key)
                                candidates.append(r)
                            if len(candidates) >= candidate_pool_size:
                                break
                    if len(candidates) >= 25:
                        break
                except Exception:
                    continue
        return candidates, primary_strategy

    raw_rows, strategy_label = fetch_candidates(enforce_collection=True)
    if not raw_rows and coll_target:
        raw_rows, strategy_label = fetch_candidates(enforce_collection=False)
        strategy_label += " (All Collections)"

    scored_results = []
    for r in raw_rows:
        m_date = str(r[0]) if r[0] else "Unknown Date"
        m_coll = str(r[1]) if r[1] else "Murli"
        m_snip = str(r[2]) if r[2] else ""
        full_content = str(r[3]) if r[3] else ""

        rich_excerpt, relevance_score = score_and_extract_passages(full_content, intent)
        parsed_sec = parse_murli_sections(full_content)

        scored_results.append({
            "date": m_date,
            "type": m_coll,
            "snippet": m_snip,
            "text": rich_excerpt if rich_excerpt else m_snip,
            "vardaan": parsed_sec.get("vardaan", ""),
            "slogan": parsed_sec.get("slogan", ""),
            "score": round(relevance_score, 1)
        })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:final_limit], strategy_label


def ask_gemini(question: str, context_chunks: list[dict], intent: dict) -> str:
    """Synthesizes a comprehensive response covering 15 to 25 distinct points with exact quotes."""
    context_blocks = []
    for idx, item in enumerate(context_chunks, 1):
        context_blocks.append(
            f"=== [Murli {idx} | Collection: {item['type']} | Date: {item['date']}] ===\n"
            f"{item['text']}\n"
        )
    formatted_context = "\n\n".join(context_blocks)

    section_focus = intent.get("section_focus")
    section_instruction = ""
    if section_focus:
        section_instruction = (
            f"\nSPECIAL FOCUS: The user is specifically interested in the '{section_focus.upper()}' section. "
            f"Prioritize quoting and explaining the {section_focus.upper()} from the provided excerpts first!"
        )

    system_instruction = (
        "You are an insightful, deeply knowledgeable spiritual research assistant for Brahma Kumaris "
        "Sakar (Daily) and Avyakt Murlis.\n\n"
        "Follow these strict principles:\n"
        "1. GROUNDED IN MURLI: Base your answer strictly on the provided Murli excerpts. Never invent dates or quotes.\n"
        "2. EXACT QUOTES (बाबा के महावाक्य): Always quote Baba's exact Hindi words from the excerpt inside blockquotes "
        "(> **\"...\"**) before explaining the point.\n"
        "3. COMPREHENSIVE SYNTHESIS (15–25 POINTS): Provide a thorough, comprehensive breakdown covering 15 to 25 distinct "
        "spiritual points/aspects found across the excerpts, categorizing them clearly under bullet points or sub-headings.\n"
        "4. CLEAR CITATIONS: Cite the exact Collection and Murli Date for every point or quote.\n"
        "5. LANGUAGE: Respond in the language the user asked in (Hindi or English)."
        + section_instruction
    )

    prompt = f"""
User Question: {question}
Core Concepts Searched: {', '.join(intent.get('core_concepts', []))}

Retrieved Murli Excerpts:
{formatted_context}

Provide a comprehensive, highly detailed response covering 15 to 25 distinct points, quoting Baba's exact words and citing the Murli dates.
"""

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )
        response = call_gemini_resilient(contents=prompt, config=config)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Search Filters UI ---
with st.expander("🔍 Search Filters", expanded=False):
    section_choice = st.selectbox(
        "Target Murli Section:",
        ["All Sections", "वरदान (Vardaan)", "स्लोगन (Slogan)", "धारणा (Dharna)", "मुख्य सार (Essence)"],
        index=0
    )

# --- Render Conversation History & Read Aloud Feature ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant":
            clean_speech_text = msg["content"].replace("*", "").replace("#", "").replace(">", "").replace("---", "")
            safe_text = json.dumps(clean_speech_text)
            
            st.markdown(f"""
            <div style="margin-top: 8px; margin-bottom: 12px;">
                <button onclick='speakText({safe_text}, "speak-btn-{idx}")' id='speak-btn-{idx}' style="background: #2b313e; color: #ffffff; border: 1px solid #41424C; border-radius: 16px; padding: 4px 12px; font-size: 13px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
                    🔊 Read Aloud
                </button>
                <button onclick='stopSpeech()' style="background: #2b313e; color: #ff4b4b; border: 1px solid #41424C; border-radius: 16px; padding: 4px 12px; font-size: 13px; cursor: pointer; margin-left: 6px; display: inline-flex; align-items: center;">
                    ⏹️ Stop
                </button>
            </div>
            """, unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander("📖 View Original Murli Excerpts, Vardaans & Snippets"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['type'].upper()} — {s['date']}**")
                    if s.get("vardaan"):
                        st.info(f"✨ **{s['vardaan']}**")
                    if s.get("slogan"):
                        st.success(f"🔔 **{s['slogan']}**")
                    st.text(s["text"][:1200] + ("..." if len(s["text"]) > 1200 else ""))
                    st.divider()

# --- Global Speech Synthesis JavaScript ---
st.markdown("""
<script>
function speakText(text, btnId) {
    if (!('speechSynthesis' in window)) {
        alert("Text-to-speech is not supported by your browser.");
        return;
    }
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'hi-IN';
    utterance.rate = 1.0;
    
    const btn = document.getElementById(btnId);
    if(btn) btn.innerText = "🔊 Speaking...";
    
    utterance.onend = function() { if(btn) btn.innerText = "🔊 Read Aloud"; };
    utterance.onerror = function() { if(btn) btn.innerText = "🔊 Read Aloud"; };
    
    window.speechSynthesis.speak(utterance);
}

function stopSpeech() {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        document.querySelectorAll("[id^='speak-btn-']").forEach(b => b.innerText = "🔊 Read Aloud");
    }
}
</script>
""", unsafe_allow_html=True)

# --- Native Streamlit Chat Input ---
active_prompt = st.chat_input("Ask question, Vardaan, or Slogan in Hindi or English...")

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Murlis and synthesizing 15-25 comprehensive points..."):
            intent = extract_search_intent(active_prompt, ui_section_override=section_choice)
            matches, strategy_used = search_turso(intent, final_limit=20)
            core_str = ", ".join(intent.get("core_concepts", []))

            if not matches:
                answer = (
                    f"No matching Murlis found for: **{core_str}**.\n\n"
                    f"💡 *Tip: Try simpler Hindi keywords or set 'Target Murli Section' to 'All Sections'.*"
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                answer = ask_gemini(active_prompt, matches, intent)
                sec_badge = f" | Section: `{intent['section_focus'].upper()}`" if intent.get("section_focus") else ""
                sources_md = (
                    f"\n\n---\n**📚 Murlis Consulted** *(Mode: {strategy_used}{sec_badge} | "
                    f"Concepts: `{core_str}`)*:\n"
                    + "\n".join([f"- **{m['type'].title()}** ({m['date']})" for m in matches])
                )
                full_response = answer + sources_md
                st.markdown(full_response)

                with st.expander("📖 View Original Murli Excerpts, Vardaans & Snippets"):
                    for s in matches:
                        st.markdown(f"**{s['type'].upper()} — {s['date']}**")
                        if s.get("vardaan"):
                            st.info(f"✨ **{s['vardaan']}**")
                        if s.get("slogan"):
                            st.success(f"🔔 **{s['slogan']}**")
                        st.text(s["text"][:1200] + ("..." if len(s["text"]) > 1200 else ""))
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": matches
                })
