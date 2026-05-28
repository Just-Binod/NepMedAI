from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os


# ── Load or build vector store 

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="jangedoo/all-MiniLM-L6-v2-nepali",
        model_kwargs={'device': 'cpu'}
    )
    if os.path.exists("./chroma_db_nepali"):
        return Chroma(persist_directory="./chroma_db_nepali", embedding_function=embeddings)

    loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splitted_data = splitter.split_documents(docs)
    return Chroma.from_documents(documents=splitted_data, embedding=embeddings, persist_directory="./chroma_db_nepali")


# ── Context retrieval ─────────

vector_store = load_vectorstore()

def get_context(query: str):
    results = vector_store.similarity_search(query=query)
    context = ""
    sources = []
    for doc in results:
        context += doc.page_content + "\n"
        book_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        entry = f"{book_name} — p.{page}"
        if entry not in sources:
            sources.append(entry)
    return context, sources


# ── Prompt ────────────────────

prompt = PromptTemplate.from_template(
    """You are NepMedAI, a bilingual medical information assistant.

LANGUAGE RULE — this is the most important rule:
- If the question is in English only → reply entirely in English
- If the question is in Nepali only → reply entirely in Nepali
- If the question mixes both languages → reply in the same mix of both languages
- Never switch language. Match exactly what the user used.

Other rules:
- Answer using ONLY the given context. Do not make up information.
- If the answer is not in context, say so clearly in the user's language.
- Be concise and friendly.
- Do NOT add any disclaimer line at the end — that is handled separately.
- Use conversation history to understand follow-up words like "their", "it", "those".

Conversation History:
{history}

Context from knowledge base:
{context}

Current Question: {question}

Answer:"""
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


# ── Ask function ──────────────

def ask(question: str, history: list):
    context, sources = get_context(question)

    # Build history string from last 6 messages (3 turns)
    recent = history[-6:]
    history_text = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    if not history_text:
        history_text = "No previous conversation."

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question, "history": history_text})
    return answer, sources


# ── UI helpers ────────────────

DISCLAIMER = (
    '<p style="color:#cc0000; font-size:10px; margin-top:6px; margin-bottom:0;">'
    '⚠️ Results may not be 100% accurate. Please consult a medical professional. &nbsp;|&nbsp; '
    'परिणामहरू १००% सही नहुन सक्छ। कृपया चिकित्सकसँग परामर्श लिनुहोस्।'
    '</p>'
)

def render_sources(sources: list):
    """Render sources in tiny muted text above the answer bubble."""
    if not sources:
        return ""
    lines = " &nbsp;·&nbsp; ".join(sources)
    return (
        f'<p style="color:#888888; font-size:11px; margin-bottom:4px; margin-top:0;">'
        f'📄 {lines}'
        f'</p>'
    )

def render_bot_message(answer: str, sources: list):
    """Render sources + answer + disclaimer as one clean block."""
    sources_html = render_sources(sources)
    # Escape for safe HTML display then wrap in a styled div
    import html
    safe_answer = html.escape(answer).replace("\n", "<br>")
    st.markdown(
        f'{sources_html}'
        f'<div style="font-size:15px; line-height:1.65; margin-bottom:2px;">{safe_answer}</div>'
        f'{DISCLAIMER}',
        unsafe_allow_html=True
    )


# ── Page config & custom CSS ──

st.set_page_config(page_title="NepMedAI", page_icon="🏥", layout="centered")

st.markdown("""
<style>
/* ── App background ── */
.stApp { background-color: #0f0f0f; }

/* ── Header ── */
.nep-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 18px;
}
.nep-logo {
    font-size: 32px;
    line-height: 1;
}
.nep-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}
.nep-subtitle {
    font-size: 12px;
    color: #888;
    margin: 0;
}
.nep-badge {
    margin-left: auto;
    background: #1a3a1a;
    color: #4caf50;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 12px;
    border: 1px solid #2d5a2d;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: #1e3a5f !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 10px 15px !important;
    display: inline-block;
    max-width: 80%;
    float: right;
    color: #e8f0fe !important;
    font-size: 15px;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 10px 15px !important;
    color: #e0e0e0 !important;
    font-size: 15px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 2px rgba(76,175,80,0.15) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4caf50 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

/* ── Welcome card ── */
.welcome-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 22px 24px;
    margin: 10px 0 22px 0;
    color: #bbb;
    font-size: 14px;
    line-height: 1.7;
}
.welcome-card h4 { color: #fff; margin-bottom: 8px; font-size: 15px; }

/* ── Suggestion chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.chip {
    background: #1e3a1e;
    border: 1px solid #2d5a2d;
    color: #81c784;
    font-size: 12px;
    padding: 5px 12px;
    border-radius: 20px;
    cursor: pointer;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────

st.markdown("""
<div class="nep-header">
  <div class="nep-logo">🏥</div>
  <div>
    <p class="nep-title">NepMedAI</p>
    <p class="nep-subtitle">Bilingual Medical Assistant &nbsp;·&nbsp; नेपाली / English</p>
  </div>
  <div class="nep-badge">● Online</div>
</div>
""", unsafe_allow_html=True)


# ── Session state ──────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Welcome card (shown only before first message) ───────────────────────────────

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
      <h4>नमस्ते! How can I help you today?</h4>
      Ask me health questions in <b>English</b>, <b>Nepali (नेपाली)</b>, or a mix of both.<br>
      I will answer in the same language you use.<br><br>
      <b>Try asking:</b>
      <div class="chip-row">
        <span class="chip">Which disease is common in Nepal?</span>
        <span class="chip">टाइफाइडका लक्षणहरू के हुन्?</span>
        <span class="chip">How to prevent malaria?</span>
        <span class="chip">बच्चाको खोकीको उपचार के हो?</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Render previous messages ───

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_bot_message(msg["content"], msg.get("sources", []))
        else:
            st.write(msg["content"])


# ── Chat input ─────────────────

if question := st.chat_input("Type your health question here... / यहाँ आफ्नो प्रश्न टाइप गर्नुहोस्..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                answer, sources = ask(question, st.session_state.messages)
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str:
                    answer = "⚠️ The AI service is currently experiencing high demand. Please wait a moment and try again.\n\nAI सेवामा अहिले धेरै अनुरोधहरू छन्। कृपया केही क्षण पर्खनुहोस् र फेरि प्रयास गर्नुहोस्।"
                else:
                    answer = f"⚠️ Something went wrong. Please try again.\n\n_(Error: {error_str})_"
                sources = []

        render_bot_message(answer, sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
