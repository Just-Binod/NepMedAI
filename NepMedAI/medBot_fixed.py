from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda  
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os


#  Load or build vector store 

# Only build once. If chroma_db folder already exists, just load it.
#        if we didn't use , it re-embedded the entire PDF every single run (very slow).

@st.cache_resource  # Streamlit caches this so it only runs once per session
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="jangedoo/all-MiniLM-L6-v2-nepali",
        model_kwargs={'device': 'cpu'}
    )

    if os.path.exists("./chroma_db_nepali"):
        # Already built before , just load it
        return Chroma(
            persist_directory="./chroma_db_nepali",
            embedding_function=embeddings
        )

    # First time: build from PDF
    # loader = PyPDFLoader("data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")

    loader = DirectoryLoader(
        "./data/",
        glob="./*.pdf",        # pick up the pdf inside the folder  also
        loader_cls=PyPDFLoader, #but using pdf loader for loading the pdf file, as DirectoryLoader was showing error
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splitted_data = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=splitted_data,
        embedding=embeddings,
        persist_directory="./chroma_db_nepali"
    )
    return vector_store


#  Context retrieval function 

# vector_store = load_vectorstore()

# def get_context(inputs: dict) -> dict:
#     query = inputs["question"]

#     results = vector_store.similarity_search(query=query)
#     context = ""
#     for doc in results:
#         context += doc.page_content + "\n"
#     return {
#         "context": context,
#         "question": query
#     }




# ////
#  Context retrieval , that  now also returns source
 
vector_store = load_vectorstore()
 
def get_context(inputs: dict) -> dict:
    query = inputs["question"]
    results = vector_store.similarity_search(query=query)
 
    context = ""
    sources = []  #  collect source info from each retrieved chunk
 
    for doc in results:
        context += doc.page_content + "\n"
 
        # Each chunk has metadata added automatically by PyPDFLoader
        raw_source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "?")
 
        # Clean up the file path to just show the filename
        book_name = os.path.basename(raw_source)
 
        # Avoid adding duplicate sources (same book + page)
        entry = f"📖 {book_name}  —  Page {page}"
        if entry not in sources:
            sources.append(entry)
 
    return {
        "context": context,
        "question": query,
        "sources": sources
    }
 






#  using Prompt 

prompt = PromptTemplate.from_template(
    """तपाईं एक सहायक हुनुहुन्छ। दिइएको सन्दर्भ प्रयोग गरी प्रश्नको उत्तर दिनुहोस्।
You are a helpful assistant. Answer questions using the given context.

Guidelines:
- Answer in the same language as the question
- Be concise but informative
- If not in context, say "I don't have information about this"
- Always add: results may not be 100% accurate, please consult a medical professional.

Context: {context}
Question: {question}"""
)


#  LLM and chain 

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# # get_context | prompt | llm; we cannot pipe a plain function.
# #         Wrapped with RunnableLambda so it works in a LangChain chain.


# # Added StrOutputParser() at the end  
# #         Without it, the chain returns an AIMessage object, not plain text.
# # so we can get without using .content

# rag_chain = RunnableLambda(get_context) | prompt | llm | StrOutputParser()



# ///////

#  LLM 
 
google_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not google_api_key:
    raise RuntimeError(
        "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    api_key=google_api_key,
)
 
 
#  Two-step ask function so we can access sources separately 
 
#  We call get_context first to grab sources, then run the chain.
#            Can't extract sources mid-pipe, so we split into two steps.
 
def ask(question: str):
    # Step 1: retrieve context + sources
    retrieved = get_context({"question": question})
 
    # Step 2: run prompt + LLM + parser
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": retrieved["context"],
        "question": retrieved["question"]
    })
 
    return answer, retrieved["sources"]
 






# /////////////////////////

# ── Streamlit UI ─────────────────────────────────────────────────────────────────
 
st.set_page_config(page_title="NepMedAI", page_icon="")
st.title(" NepMedAI : bilingual Medical Assistant")
st.caption("Ask health questions in Nepali or English")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # CHANGE 4: Re-show sources in chat history
        if msg.get("sources"):
            with st.expander(" Sources referred"):
                for s in msg["sources"]:
                    st.write(s)
 
# Chat input
if question := st.chat_input("Ask a health question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Searching medical knowledge..."):
            try:
                answer, sources = ask(question)
            except Exception as e:
                # Gemini returns a 503 when the API is overloaded.
                # Show a friendly message instead of a scary traceback.
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str:
                    answer = "⚠️ The AI service is currently experiencing high demand. Please wait a moment and try again."
                else:
                    answer = f"⚠️ Something went wrong. Please try again.\n\n_(Error: {error_str})_"
                sources = []
 
        st.write(answer)
 
        #  Show sources in a collapsible expander below the answer
        if sources:
            with st.expander(" Sources referred"):
                for s in sources:
                    st.write(s)
 
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })