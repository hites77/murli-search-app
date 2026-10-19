import os
import sqlite3
import streamlit as st
import gdown
from google import genai

# Page Configuration for Mobile
st.set_page_config(
    page_title="Murli Search AI",
    page_icon="🕊️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🕊️ Murli Search Assistant")
st.caption("Ask questions across 8,200+ discourses in Hindi, English, or Hinglish.")

# Database Configuration
DB_PATH = "murli_search.db"
# Replace with your actual Google Drive File ID from Step 1:
GDRIVE_FILE_ID = "1AOvN9VzhJ7X3R-TKWDTXHOyCWTIGt6El"

@st.cache_resource(show_spinner=False)
def ensure_database():
    """Downloads the database from Google Drive if not present on server disk."""
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 100_000_000:
        with st.spinner("Downloading Murli database archive (~400MB, one-time setup)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, DB_PATH, quiet=False)

# Ensure database is available
ensure_database()

# Retrieve API Key from Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY. Please add it to your Streamlit App Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"


def extract_keywords(user_query: str) -> str:
    """Uses Gemini to extract core Hindi Devanagari root keywords."""
    if len(user_query.split()) <= 2 and any('\u0900' <= c <= '\u097F' for c in user_query):
        return user_query.strip()

    prompt = f"""
Given this question or topic: "{user_query}"
Extract 1 to 2 core spiritual keywords in Devanagari Hindi as used in Brahma Kumaris Murlis.
Output ONLY the keywords in Hindi (no punctuation, no explanation, no english).
Examples:
- "When does Satyug begin?" -> सत्ययुग
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


def search_database(search_term: str, limit: int = 5) -> list[dict]:
    """Queries SQLite FTS5 trigram index."""
    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

    try:
        formatted = f'"{search_term}"' if " " in search_term else search_term
        cursor.execute(sql, (formatted, limit))
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        terms = search_term.split()
        if len(terms) > 1:
            try:
                cursor.execute(sql, (" OR ".join(terms), limit))
                rows = cursor.fetchall()
            except Exception:
                rows = []
        else:
            rows = []

    conn.close()

    results = []
    for date, coll, snip, full_content in rows:
        results.append({
            "date": date or "Unknown Date",
            "type": coll or "Murli",
            "snippet": snip,
            "text": full_content[:2500] if full_content else snip
        })
    return results


def ask_gemini(question: str, context_chunks: list[dict]) -> str:
    """Builds grounded prompt with retrieved excerpts."""
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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.2,
        }
    )
    return response.text


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Type your question or phrase..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching archive..."):
            keywords = extract_keywords(prompt)
            matches = search_database(keywords, limit=5)

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
