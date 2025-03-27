####   More descriptive   


import streamlit as st
import os
import time
import re
import tempfile
from dotenv import load_dotenv

# Import LangChain components to handle document ingestion, semantic retrieval,
# and advanced language model interactions.
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------------------------------------------------------
# Environment Setup & System Constants
# -----------------------------------------------------------------------------

# Load environment variables from a .env file (e.g., API keys)
load_dotenv()

# Define constants for embedding and language model configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama3-70b-8192"
CHUNK_SIZE = 1500         # Maximum size for each document chunk
CHUNK_OVERLAP = 300       # Overlap between consecutive chunks for context continuity

# -----------------------------------------------------------------------------
# Initialization Functions
# -----------------------------------------------------------------------------

@st.cache_resource
def load_embeddings():
    """
    Load the HuggingFace embeddings model, configured to run on CPU.
    This model converts text into vector embeddings for semantic search.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_llm():
    """
    Initialize the ChatGroq LLM with parameters optimized for precise answers.
    A lower temperature ensures the output remains focused and technical.
    """
    return ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name=LLM_MODEL,
        temperature=0.3,
        max_tokens=1200,
        top_p=0.85
    )

def format_response(text: str) -> str:
    """
    Format the LLM response for readability and consistency:
    
    - Replace hyphenated lists with bullet points.
    - Ensure Python code blocks are correctly formatted.
    - Normalize citation markers (e.g., [Page X]) for traceability.
    """
    text = re.sub(r'\n- ', '\n• ', text)
    text = re.sub(r'```python\s*(.*?)\s*```', r'```python\n\1\n```', text, flags=re.DOTALL)
    text = re.sub(r'\[ ?Page (\d+) ?\]', r'[Page \1]', text)
    return text.strip()

# -----------------------------------------------------------------------------
# Load Core Components and Initialize Session State
# -----------------------------------------------------------------------------

embeddings = load_embeddings()  # Embedding model for document semantic mapping
llm = load_llm()                # Language model for generating analysis responses

# Session state variables to maintain conversation history and document context
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# -----------------------------------------------------------------------------
# Streamlit Layout & Document Processing Sidebar
# -----------------------------------------------------------------------------

st.title("📄 Enterprise Document Analysis Suite")

with st.sidebar:
    st.header("Document Configuration")
    # Allow the user to upload a PDF document
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    # Option to maintain conversation context for follow-up queries
    remember_history = st.checkbox("Maintain Conversation Context", value=True)
    
    if uploaded_file and not st.session_state.processed:
        if st.button("Process Document"):
            with st.spinner("Performing Document Analysis..."):
                try:
                    # Temporarily save the uploaded PDF file for processing
                    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                        temp_pdf.write(uploaded_file.getbuffer())
                        temp_file_path = temp_pdf.name

                    # Use PyPDFLoader to load the document and extract its pages
                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()
                    
                    # Update document metadata with source and page information for citations
                    for doc in docs:
                        doc.metadata.update({
                            "source": uploaded_file.name,  # Document name as source [Page 1]
                            "page": doc.metadata.get("page", 1),
                            "total_pages": len(docs)
                        })

                    # Split the document into manageable semantic units
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        separators=["\n\n", "\n", ". ", " "],
                        add_start_index=True
                    )
                    final_docs = text_splitter.split_documents(docs)

                    # Create a vector store for efficient semantic retrieval
                    st.session_state.vectors = FAISS.from_documents(
                        documents=final_docs,
                        embedding=embeddings
                    )
                    st.session_state.processed = True
                    st.success(f"Analysis Complete: Processed {len(final_docs)} semantic units")
                    os.remove(temp_file_path)

                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")
                    st.session_state.processed = False

# -----------------------------------------------------------------------------
# System Prompt for Technical Analysis
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """Act as a technical expert analyzing enterprise documents. Your answer must:

1. Present a **clear, structured response** with descriptive section headings.
2. Provide **direct document grounding** by citing sources using [Page X] references.
3. Include **self-contained code examples** formatted as ```python code blocks``` for clarity.
4. Offer **context-aware technical depth** with definitions, examples, and implementation insights.
5. Maintain a **professional yet accessible tone**.

Conversation History:
{history}

Document Context:
{context}

Question:
{input}"""

# -----------------------------------------------------------------------------
# Chat Interface: User Query & LLM Response
# -----------------------------------------------------------------------------

st.header("Technical Analysis Interface")
# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter technical query..."):
    if not st.session_state.processed:
        st.error("Please process a document first")
        st.stop()

    # Save the user's query in the session history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Retrieve relevant segments from the processed document using semantic search
            retriever = st.session_state.vectors.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 8, 'fetch_k': 25, 'lambda_mult': 0.5}
            )

            # Create a chain with the system prompt for structured technical responses
            prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            chain = create_retrieval_chain(retriever, document_chain)

            # Optionally include recent conversation history to maintain context
            history_context = ""
            if remember_history:
                history_context = "\n".join(
                    [f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages[-4:]]
                )

            # Invoke the chain to generate a response from the LLM
            start_time = time.time()
            response = chain.invoke({
                "input": prompt,
                "history": history_context
            })

            # Format the raw response to ensure clarity and consistency
            raw_answer = response["answer"]
            answer = format_response(raw_answer)
            st.markdown(answer)
            st.caption(f"Response Generated in {time.time() - start_time:.2f}s")

            # Display detailed source verification for traceable citations
            with st.expander("Source Verification"):
                for i, doc in enumerate(response["context"]):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    st.write(f"**{source} - Page {page}**")
                    st.code(doc.page_content, language='text')
                    st.divider()

            # Append the assistant's answer to the conversation history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

# -----------------------------------------------------------------------------
# Session Control Panel: Manage Conversation & Document Context
# -----------------------------------------------------------------------------

with st.container():
    st.write("Session Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize New Session"):
            st.session_state.messages = []
            st.experimental_rerun()
    with col2:
        if st.button("Clear Document Context"):
            st.session_state.vectors = None
            st.session_state.processed = False
            st.session_state.messages = []
            st.experimental_rerun()
            
            
            
