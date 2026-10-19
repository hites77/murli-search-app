import streamlit as st
import libsql_client
from google import genai
from google.genai import types

# --- Mobile Page Configuration ---
st.set_page_config(
    page_title="Murli Search AI",
    page_icon="🕊️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🕊️ Murli Search Assistant")
st.caption("Ask questions across 8,200+ discourses in Hindi, English, or Hinglish.")

# --- Retrieve Credentials ---
gemini_key = st.secrets.get("GEMINI_API_KEY")
turso_url = st.secrets.get("TURSO_DATABASE_URL")
turso_token = st.secrets.get("TURSO_AUTH_TOKEN")

if not (gemini_key and turso_url and turso_token):
    st.error("Missing configuration. Please verify GEMINI_API_KEY, TURSO_DATABASE_URL, and TURSO_AUTH_TOKEN in Streamlit Secrets.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=gemini_key.strip())
MODEL_NAME = "gemini-2.5-flash"


def get_turso_client():
    """Connects to Turso via HTTPS for serverless environments."""
    http_url = turso_url.replace("libsql://", "https://")
    return libsql_client.create_client_sync(url=http_url, auth_token=turso_token.strip())


def extract_keywords(user_query: str) -> str:
    """Extracts 1 to 2 core Hindi root keywords for FTS5 lookup."""
    # If the user typed a short Hindi query directly, use it as-is
    if len(user_query.split()) <= 2 and any('\u0900' <= c <= '\u097F' for c in user_query):
        return user_query.strip()

    prompt = f"""
Given this question or topic: "{user_query}"
Extract 1 to 2 core spiritual keywords in Devanagari Hindi as used in Brahma Kumaris Murlis.
Output ONLY the keywords in Hindi (no punctuation, no explanation, no english).
Examples:
- "When does Satyug begin?" -> सत्ययुग
- "sangamyug" -> संगमयुग
- "how to stay detached and observe" -> साक्षी दृष्टा
- "मन को कैसे शांत करें" -> मन एकाग्र
"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip().replace('"', '').replace("'", "")
    except Exception:
        return user_query


def search_turso(search_term: str, limit: int = 5) -> list[dict]:
    """Queries the remote Turso FTS5 trigram index."""
    sql = """
        SELECT 
            m.murli_date,
            m.collection,
            snippet(murlis_fts, 0, '>>>', '<<<', '...', 40) AS match_snippet,
            m.content
        FROM murlis_fts
        JOIN murlis m ON m.id = murlis_fts.rowid
        WHERE murlis_fts MATCH ?
        ORDER BY rank
        LIMIT ?;
    """
    formatted = f'"{search_term}"' if " " in search_term else search_term

    rows = []
    try:
        with get_turso_client() as db:
            result = db.execute(sql, [formatted, limit])
            rows = result.rows
    except Exception:
        terms = search_term.split()
        if len(terms) > 1:
            try:
                with get_turso_client() as db:
                    result = db.execute(sql, [" OR ".join(terms), limit])
                    rows = result.rows
            except Exception:
                rows = []

    results = []
    for r in rows:
        full_content = str(r[3]) if r[3] else ""
        results.append({
            "date": str(r[0]) if r[0] else "Unknown Date",
            "type": str(r[1]) if r[1] else "Murli",
            "snippet": str(r[2]) if r[2] else "",
            "text": full_content[:2500] if full_content else str(r[2])
        })
    return results


def ask_gemini(question: str, context_chunks: list[dict]) -> str:
    """Builds grounded prompt with retrieved excerpts using standard SDK config."""
    context_blocks = []
    for idx, item in enumerate(context_chunks, 1):
        context_blocks.append(
            f"--- [Document {idx}] ---\n"
            f"Collection: {item['type']}\n"
            f"Date: {item['date']}\n"
            f"Excerpt:\n{item['text']}\n"
        )
    formatted_context = "\n".join(context_blocks)

    system_instruction = (
        "You are an assistant specialized in Brahma Kumaris Daily and Avyakt Murli literature. "
        "Answer the user's question strictly using only the provided context excerpts. "
        "Always cite the Murli date and collection type for each point you mention. "
        "If the answer cannot be determined from the excerpts, state clearly that the matching records "
        "do not contain that specific detail. You may respond in Hindi or English depending on the user's query."
    )

    prompt = f"""
Provided Murli Excerpts:
{formatted_context}

User Question: {question}
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


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question or phrase..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching cloud archive..."):
            keywords = extract_keywords(prompt)
            matches = search_turso(keywords, limit=5)

            if not matches:
                answer = f"No direct records found for: **{keywords}**. Try searching with related spiritual terms like *सत्ययुग*, *अमृतवेला*, or *संतुष्टता*."
            else:
                answer = ask_gemini(prompt, matches)
                sources_md = "\n\n---\n**Sources Consulted:**\n" + "\n".join(
                    [f"- *{m['type']}* ({m['date']})" for m in matches]
                )
                answer += sources_md

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
