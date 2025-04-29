
import streamlit as st
import os
import re
import time
from typing import List, Dict, Optional, Tuple
import tempfile
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class Config:
    """Optimized configuration for fast RAG pipeline"""
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    LLM_MODEL = "mixtral-8x7b-32768"
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 128
    INITIAL_RETRIEVE = 15
    FINAL_RERANK = 3
    MAX_HISTORY = 4
    MAX_RESPONSE_TOKENS = 1024
    RERANK_WORKERS = 4

class FastDocumentProcessor:
    """Optimized document processing with parallelization"""
    
    @staticmethod
    def process_pdf(uploaded_file) -> List[Document]:
        """Fast PDF processing with parallel text extraction"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            temp_path = temp_pdf.name
        
        try:
            with ThreadPoolExecutor() as executor:
                loader = PyPDFLoader(temp_path)
                raw_docs = loader.load()
                processed_docs = list(executor.map(FastDocumentProcessor.process_page, raw_docs))
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            return text_splitter.split_documents(processed_docs)
        finally:
            try:
                os.remove(temp_path)
            except:
                pass
    
    @staticmethod
    def process_page(doc: Document) -> Document:
        """Process a single page with optimized text cleaning"""
        content = re.sub(r'\s+', ' ', doc.page_content).strip()
        content = re.sub(r'-\n', '', content)
        return Document(
            page_content=content,
            metadata={
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0) + 1,
                "char_count": len(content)
            }
        )


class FastReranker:
    """Optimized reranking implementation"""
    
    def __init__(self):
        self.model = CrossEncoder(Config.RERANKER_MODEL)
    
    def parallel_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Parallelized document reranking with type checking"""
        if not documents:
            return []
        
        # Validate document types
        valid_docs = []
        for doc in documents:
            if isinstance(doc, Document):
                valid_docs.append(doc)
            else:
                st.error("Invalid document type detected in reranking")
                continue
        
        pairs = [(query, doc.page_content) for doc in valid_docs]
        
        batch_size = 32
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            scores.extend(self.model.predict(batch, show_progress_bar=False))
        
        scored_docs = sorted(zip(valid_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:Config.FINAL_RERANK]]


class FastAIComponents:
    """Optimized AI components with caching"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.reranker = FastReranker()
        self.llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name=Config.LLM_MODEL,
            temperature=0.3,
            max_tokens=Config.MAX_RESPONSE_TOKENS,
            top_p=0.8
        )

class FastResponseGenerator:
    """Optimized response generation pipeline"""
    
    SYSTEM_PROMPT = """You are Yeva, an AI assistant. Follow these rules:

1. Respond in under 3 sentences when possible
2. Use bullet points • for lists
3. Cite sources as [Page X]
4. Use **bold** for important terms

Context: {context}
History: {history}
Query: {input}"""

    @staticmethod
    def format_response(text: str) -> str:
        """Optimized response formatting"""
        replacements = [
            (r'\[ ?Page (\d+) ?\]', r'[Page \1]'),
            (r'(?<!\n)\n(?!\n)', ' '),
            (r'\n- ', '\n• ')
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        return text.strip()
    
    @staticmethod
    def generate_history(messages: List[Dict]) -> str:
        """Optimized history generation"""
        return "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Yeva'}: {msg['content'][:200]}"
            for msg in messages[-Config.MAX_HISTORY:-1]
        )

def initialize_session_state():
    """Initialize all required session state variables"""
    required_state = {
        "messages": [{
            "role": "assistant",
            "content": "👋 Hi! I'm Yeva,AI assistant. Ask me anything!"
        }],
        "vector_store": None,
        "document_processed": False,
        "last_query": None
    }
    
    for key, value in required_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    """Main application entry point"""
    # Initialize session state first
    initialize_session_state()
    
    # Initialize components
    ai_components = FastAIComponents()
    
    # Setup page
    st.set_page_config(
        page_title="Yeva - AI Assistant",
        page_icon="⚡",
        layout="centered"
    )
    st.title("⚡ Yeva - AI Assistant")
    st.caption("Instant answers about Yeppar's immersive technologies")
    
    # Sidebar
    with st.sidebar:
        st.header("⚡ Quick Upload")
        uploaded_file = st.file_uploader(
            "Drag PDF here", 
            type=["pdf"],
            help="Upload technical documents for faster, accurate answers"
        )
        
        if uploaded_file and not st.session_state.document_processed:
            with st.spinner("Super fast processing..."):
                try:
                    split_docs = FastDocumentProcessor.process_pdf(uploaded_file)
                    st.session_state.vector_store = FAISS.from_documents(
                        split_docs,
                        ai_components.embeddings
                    )
                    st.session_state.document_processed = True
                    st.success(f"Ready! Processed {len(split_docs)} chunks in seconds")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📄 Loaded {uploaded_file.name} with {len(split_docs)} sections. Ask me anything about it!"
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="⚡" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about XR tech..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_query = time.time()
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Thinking fast..."):
                start_time = time.time()
                
                # Process query
                context = []
                sources = []
                if st.session_state.vector_store:
                    docs = st.session_state.vector_store.similarity_search(
                        prompt, 
                        k=Config.INITIAL_RETRIEVE
                    )
                    if docs:
                        reranked_docs = ai_components.reranker.parallel_rerank(prompt, docs)
                        context = [doc.page_content for doc in reranked_docs]
                        sources = [str(doc.metadata.get("page", "")) for doc in reranked_docs if doc.metadata.get("page")]
                
                # Generate response
                history = FastResponseGenerator.generate_history(st.session_state.messages)
                prompt_template = ChatPromptTemplate.from_template(FastResponseGenerator.SYSTEM_PROMPT)
                chain = create_stuff_documents_chain(ai_components.llm, prompt_template)
                
                response = chain.invoke({
                    "input": prompt,
                    "context": context,
                    "history": history
                })
                formatted_response = FastResponseGenerator.format_response(response)
                
                # Stream response
                response_placeholder = st.empty()
                full_response = ""
                for chunk in formatted_response.split():
                    full_response += chunk + " "
                    response_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)
                response_placeholder.markdown(full_response)
                
                if sources:
                    st.caption(f"Sources: {', '.join(sources)}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
    
    st.divider()
    st.caption( Yeva - Fastest XR Assistant ")

if __name__ == "__main__":
    main()
