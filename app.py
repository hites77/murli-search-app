import hashlib
import json
import re
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
st.caption("Semantic + Section-Aware Search across 8,200+ Sakar & Avyakt Murlis.")

# --- Retrieve & Validate Secrets ---
gemini_key = st.secrets.get("GEMINI_API_KEY")
turso_url = st.secrets.get("TURSO_DATABASE_URL")
turso_token = st.secrets.get("TURSO_AUTH_TOKEN")

if not (gemini_key and turso_url and turso_token):
    st.error("Missing configuration. Please check GEMINI_API_KEY, TURSO_DATABASE_URL, and TURSO_AUTH_TOKEN in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=gemini_key.strip())
MODEL_NAME = "gemini-3.8-flash"

# --- Section Marker Patterns (Upgrade 2) ---
SECTION_MARKERS = {
    "vardaan": ["वरदान:", "वरदान :-", "वरदान-", "वरदान :", "वरदान"],
    "slogan": ["स्लोगन:", "स्लोगन :-", "स्लोगन-", "स्लोगन :", "स्लोगन"],
    "dharna": ["धारणा के लिए मुख्य सार", "धारणा के लिए", "धारणा:-", "धारणा :"],
    "saar": ["मीठे बच्चे", "प्रातःमुरली", "मुख्य सार"]
}


def get_turso_client():
    """Connects to Turso via stateless HTTPS."""
    http_url = turso_url.strip().replace("libsql://", "https://")
    return libsql_client.create_client_sync(url=http_url, auth_token=turso_token.strip())


def transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    """Uses Gemini multimodal audio processing to transcribe spoken queries."""
    prompt = (
        "Listen to this audio and transcribe the spoken Hindi, English, or Hinglish question accurately. "
        "It is a question about Brahma Kumaris Murli concepts (e.g., Avyakt, Sakar, Baba, Vardaan, Slogan, Dharna, Pavitrata, Yog). "
        "Prefer Devanagari Hindi script for Hindi words. Return ONLY the plain transcribed text."
    )
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                prompt
            ]
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Voice Transcription Error: {e}")
        return ""


def extract_search_intent(user_query: str, ui_section_override: str = "All") -> dict:
    """
    UPGRADE 1 & 2:
    Extracts core concepts, BK spiritual metaphors/Yuktis (Semantic Expansion),
    and detects specific Murli sections (Vardaan, Slogan, Dharna, Saar).
    """
    prompt = f"""
You are a senior Brahma Kumaris Murli scholar. Analyze this user query: "{user_query}"

Return a valid JSON object with these exact keys:
1. "collection": "avyakt" (if user asks for Avyakt Murli / BapDada), "daily" (if Sakar / Daily Murli), or null.
2. "section_focus": "vardaan" (if asking for a Vardaan / blessing), "slogan" (if asking for a Slogan), "dharna" (if asking for Dharna points), "saar" (if asking for Mukhya Saar / essence), or null.
3. "date_filter": string (e.g., "1979", "2022-07", "18-01") if a specific year or date is mentioned, else null.
4. "core_concepts": list of 1 to 3 essential Hindi words/roots in Devanagari (e.g., "पवित्रता", "मास्टर", "सर्वशक्ति"). Do NOT include generic words like बाबा, मुरली, वरदान, स्लोगन, कहा, क्या, कैसे.
5. "synonyms": list of 2 to 4 alternative Hindi spellings or closely related Murli terms in Devanagari (e.g., "सर्वशक्तिवान", "सर्वशक्तिमान", "पवित्र").
6. "semantic_yuktis": list of 2 to 3 deeper BK Murli metaphors or Yuktis Baba uses for this topic in Devanagari (e.g., if user asks about obstacles/fear -> ["विघ्न", "पहाड़", "साक्षी दृष्टा", "अचल"]; if purity/power -> ["फाउंडेशन", "ब्रह्मचर्य", "विजयी रत्न"]).

Output ONLY valid JSON.
"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        data = json.loads(response.text.strip())
    except Exception:
        stop_words = {"किस", "अव्यक्त", "साकार", "मुरली", "में", "बाबा", "ने", "कहा", "है", "कि", "से", "ही", "बनते", "क्या", "कैसे", "वरदान", "स्लोगन"}
        words = [w.strip() for w in user_query.split() if w.strip() not in stop_words and len(w.strip()) > 2]
        data = {
            "collection": "avyakt" if ("अव्यक्त" in user_query or "avyakt" in user_query.lower()) else None,
            "section_focus": None,
            "date_filter": None,
            "core_concepts": words[:3] if words else [user_query.strip()],
            "synonyms": [],
            "semantic_yuktis": []
        }

    # Apply UI Filter Override if user selected a specific chip
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
    """
    UPGRADE 2: Automatically isolates Mukhya Saar, Dharna, Vardaan, and Slogan 
    from the raw Murli text so Gemini can see exact structured sections.
    """
    if not full_text:
        return {"saar": "", "dharna": "", "vardaan": "", "slogan": ""}

    sections = {
        "saar": full_text[:1000].strip(),
        "dharna": "",
        "vardaan": "",
        "slogan": ""
    }

    # Search in the bottom 45% of the Murli where Dharna, Vardaan, and Slogan appear
    tail_start = max(0, int(len(full_text) * 0.55))
    tail_text = full_text[tail_start:]

    # Locate Dharna
    for marker in SECTION_MARKERS["dharna"]:
        idx = tail_text.find(marker)
        if idx != -1:
            sections["dharna"] = tail_text[idx:idx + 1200].strip()
            break

    # Locate Vardaan
    for marker in SECTION_MARKERS["vardaan"]:
        idx = tail_text.rfind(marker)
        if idx != -1:
            sections["vardaan"] = tail_text[idx:idx + 900].strip()
            break

    # Locate Slogan
    for marker in SECTION_MARKERS["slogan"]:
        idx = tail_text.rfind(marker)
        if idx != -1:
            sections["slogan"] = tail_text[idx:idx + 400].strip()
            break

    return sections


def score_and_extract_passages(full_text: str, intent: dict, window_chars: int = 950, max_windows: int = 3) -> tuple[str, float]:
    """
    UPGRADE 1 & 2:
    Scores a Murli based on:
    - Core keyword hits (+5 pts per unique core term, +1.5 per frequency)
    - Synonym hits (+3 pts)
    - Semantic Yukti hits (+2 pts)
    - Section Focus bonus (+8 pts if terms appear inside the requested section like Vardaan/Slogan)
    Extracts the highest-density windows plus the structured Vardaan/Slogan footer.
    """
    if not full_text:
        return "", 0.0

    core = [t for t in intent.get("core_concepts", []) if t]
    syns = [t for t in intent.get("synonyms", []) if t]
    yuktis = [t for t in intent.get("semantic_yuktis", []) if t]
    section_focus = intent.get("section_focus")

    all_search_terms = list(dict.fromkeys(core + syns + yuktis))
    sections = parse_murli_sections(full_text)

    # Calculate semantic relevance score
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

    # Check if user wants a specific section (Vardaan / Slogan / Dharna / Saar)
    focused_section_text = ""
    if section_focus and sections.get(section_focus):
        focused_section_text = sections[section_focus]
        score += 4.0
        for term in all_search_terms:
            if term in focused_section_text:
                score += 8.0  # Huge boost if the keyword is inside the requested Vardaan/Slogan!

    # Break Murli into overlapping chunks to find the highest-density spiritual paragraphs
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
        # Proximity bonus when multiple distinct concepts appear in the same paragraph!
        if unique_hits >= 2:
            c_score += unique_hits * 4.0
        if c_score > 0:
            scored_chunks.append((c_score, start, end, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Pick top non-overlapping chunks
    selected_Passages = []
    used_ranges = []
    for c_score, s, e, chunk in scored_chunks:
        if any(not (e < us or s > ue) for us, ue in used_ranges):
            continue
        used_ranges.append((s, e))
        prefix = "..." if s > 0 else ""
        suffix = "..." if e < len(full_text) else ""
        selected_Passages.append(f"{prefix}{chunk.strip()}{suffix}")
        if len(selected_Passages) >= max_windows:
            break

    # Add top proximity bonus to overall document score
    if scored_chunks:
        score += scored_chunks[0][0]

    # Assemble final context block for this Murli
    blocks = []
    if focused_section_text:
        blocks.append(f"🎯 [TARGETED SECTION — {section_focus.upper()}]:\n{focused_section_text}")

    if selected_Passages:
        blocks.append("📜 [MATCHING MURLI PARAGRAPHS]:\n" + "\n\n---\n\n".join(selected_Passages))
    else:
        blocks.append(full_text[:1800])

    # Always attach Vardaan & Slogan at the bottom if present and not already duplicated
    footer_parts = []
    if sections["vardaan"] and section_focus != "vardaan":
        footer_parts.append(f"✨ [वरदान / Vardaan]: {sections['vardaan'][:450]}")
    if sections["slogan"] and section_focus != "slogan":
        footer_parts.append(f"🔔 [स्लोगन / Slogan]: {sections['slogan'][:250]}")
    if footer_parts:
        blocks.append("\n".join(footer_parts))

    return "\n\n".join(blocks), score


def search_turso(intent: dict, final_limit: int = 6) -> tuple[list[dict], str]:
    """
    Two-Stage Retrieval:
    Stage 1: Cascading FTS5 search retrieves up to 15 candidate Murlis from Turso.
    Stage 2: Python Semantic & Section Re-Ranker scores all 15 candidates and returns top `final_limit`.
    """
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

    # Tier 0 (Section-Specific): If user asked for a Vardaan/Slogan/Dharna, search for the section header AND concepts
    if section_term and core:
        core_or_syn = " OR ".join([f'"{t}"' for t in (core + syns)])
        fts_tiers.append((f"Section-Targeted ({section_term})", f'"{section_term}" AND ({core_or_syn})'))

    # Tier 1: All core concepts (Strict AND)
    if len(core) >= 2:
        fts_tiers.append(("Strict Concept Match", " AND ".join([f'"{c}"' for c in core])))

    # Tier 2: First core concept + any synonym or semantic yukti
    if len(core) >= 1 and len(all_terms) >= 2:
        others = " OR ".join([f'"{t}"' for t in all_terms[1:]])
        fts_tiers.append(("Contextual + Synonym Match", f'"{core[0]}" AND ({others})'))

    # Tier 3: Pairwise combinations of top 3 terms
    if len(core) >= 3:
        pair_q = f'("{core[0]}" AND "{core[1]}") OR ("{core[0]}" AND "{core[2]}") OR ("{core[1]}" AND "{core[2]}")'
        fts_tiers.append(("Pairwise Concept Match", pair_q))

    # Tier 4: Broad Semantic + Yukti OR match
    fts_tiers.append(("Semantic & Yukti Expansion", " OR ".join([f'"{t}"' for t in all_terms])))

    coll_target = intent.get("collection")
    date_target = intent.get("date_filter")
    candidate_pool_size = 15  # Fetch 15 candidates for Stage-2 semantic re-ranking

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
                    if len(candidates) >= 8:
                        break
                except Exception:
                    continue
        return candidates, primary_strategy

    raw_rows, strategy_label = fetch_candidates(enforce_collection=True)

    if not raw_rows and coll_target:
        raw_rows, strategy_label = fetch_candidates(enforce_collection=False)
        strategy_label += " (All Collections)"

    # --- STAGE 2: Semantic & Section Re-Ranking ---
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

    # Sort candidates by semantic + section relevance score (highest first)
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    return scored_results[:final_limit], strategy_label


def ask_gemini(question: str, context_chunks: list[dict], intent: dict) -> str:
    """Synthesizes a deeply grounded response quoting Baba's exact words and section references."""
    context_blocks = []
    for idx, item in enumerate(context_chunks, 1):
        context_blocks.append(
            f"=== [Murli {idx} | Collection: {item['type']} | Date: {item['date']} | Relevance Score: {item['score']}] ===\n"
            f"{item['text']}\n"
        )
    formatted_context = "\n\n".join(context_blocks)

    section_focus = intent.get("section_focus")
    section_instruction = ""
    if section_focus:
        section_instruction = (
            f"\nSPECIAL FOCUS: The user is specifically interested in the '{section_focus.upper()}' section "
            f"(e.g., Vardaan / Slogan / Dharna). Prioritize quoting and explaining the {section_focus.upper()} "
            f"from the provided excerpts first!"
        )

    system_instruction = (
        "You are an insightful, deeply knowledgeable spiritual research assistant for Brahma Kumaris "
        "Sakar (Daily) and Avyakt Murlis.\n\n"
        "Follow these strict principles:\n"
        "1. GROUNDED IN MURLI: Base your answer strictly on the provided Murli excerpts. Never invent dates or quotes.\n"
        "2. EXACT QUOTES (बाबा के महावाक्य): Always quote Baba's exact Hindi words from the excerpt inside blockquotes "
        "(> **\"...\"**) before explaining the point.\n"
        "3. CLEAR CITATIONS: Cite the exact Collection (Sakar/Daily or Avyakt), Murli Date, and Section (e.g., मुख्य सार, धारणा, वरदान, स्लोगन) for every point.\n"
        "4. SPIRITUAL DEPTH: Connect related spiritual expressions naturally (e.g., पवित्रता -> मास्टर सर्वशक्तिवान/सर्वशक्तिमान, "
        "विघ्न -> पहाड़ को राई, साक्षी दृष्टा). Explain Baba's practical Yukti (method for Purusharth) clearly.\n"
        "5. LANGUAGE: Respond in the language the user asked in (Hindi or English)."
        + section_instruction
    )

    prompt = f"""
User Question: {question}
Core Concepts Searched: {', '.join(intent.get('core_concepts', []))}
Semantic Yuktis Searched: {', '.join(intent.get('semantic_yuktis', []))}

Retrieved Murli Excerpts:
{formatted_context}

Provide a clear, spiritually accurate, well-structured response quoting Baba's exact words and citing the Murli dates.
"""

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

# --- Section Filter UI (Upgrade 2) ---
with st.expander("🔍 Search Filters & Voice Input", expanded=False):
    col1, col2 = st.columns([1, 1])
    with col1:
        section_choice = st.selectbox(
            "Target Murli Section:",
            ["All Sections", "वरदान (Vardaan)", "स्लोगन (Slogan)", "धारणा (Dharna)", "मुख्य सार (Essence)"],
            index=0
        )
    with col2:
        voice_file = st.audio_input("🎙️ Record Voice Query")

# --- Render Conversation History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 View Original Murli Excerpts, Vardaans & Snippets"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['type'].upper()} — {s['date']}** *(Match Score: {s.get('score', '-')})*")
                    if s.get("vardaan"):
                        st.info(f"✨ **{s['vardaan']}**")
                    if s.get("slogan"):
                        st.success(f"🔔 **{s['slogan']}**")
                    st.text(s["text"][:1200] + ("..." if len(s["text"]) > 1200 else ""))
                    st.divider()

text_prompt = st.chat_input("Ask any Murli question, Vardaan, or Slogan...")

active_prompt = None

# Process voice input ONLY if it is a new recording
if voice_file is not None:
    audio_bytes = voice_file.read()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("🎙️ Transcribing your voice query..."):
            mime = voice_file.type or "audio/wav"
            transcribed = transcribe_audio(audio_bytes, mime)
            if transcribed:
                active_prompt = transcribed

if text_prompt:
    active_prompt = text_prompt

# --- Execute Search & Answer Generation ---
if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Expanding spiritual concepts, checking sections & ranking Murlis..."):
            intent = extract_search_intent(active_prompt, ui_section_override=section_choice)
            matches, strategy_used = search_turso(intent, final_limit=6)

            core_str = ", ".join(intent.get("core_concepts", []))
            yukti_str = ", ".join(intent.get("semantic_yuktis", []))

            if not matches:
                answer = (
                    f"No matching Murlis found for: **{core_str}**.\n\n"
                    f"💡 *Tip: Try simpler Hindi keywords like `पवित्रता सर्वशक्तिमान` or set 'Target Murli Section' to 'All Sections'.*"
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                answer = ask_gemini(active_prompt, matches, intent)
                sec_badge = f" | Section: `{intent['section_focus'].upper()}`" if intent.get("section_focus") else ""
                sources_md = (
                    f"\n\n---\n**📚 Murlis Consulted** *(Mode: {strategy_used}{sec_badge} | "
                    f"Concepts: `{core_str}` | Semantic Yuktis: `{yukti_str}`)*:\n"
                    + "\n".join([f"- **{m['type'].title()}** ({m['date']}) — *Score: {m['score']}*" for m in matches])
                )
                full_response = answer + sources_md
                st.markdown(full_response)

                with st.expander("📖 View Original Murli Excerpts, Vardaans & Snippets"):
                    for s in matches:
                        st.markdown(f"**{s['type'].upper()} — {s['date']}** *(Match Score: {s['score']})*")
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
