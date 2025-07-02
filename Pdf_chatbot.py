import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
import os
import tempfile
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TEXT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    GROQ_MODEL = "llama3-70b-8192"
    TEXT_TOP_K = 5
    MAX_TOKENS = 500

@st.cache_resource
def load_models():
    return {
        "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL),
        "text_splitter": RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        ),
        "groq_client": Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    }

class DocumentProcessor:
    def __init__(self):
        self.models = load_models()
        
    def process_pdf(self, file_bytes):
        with st.spinner("Analyzing document..."):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            texts = self._extract_text(doc)
            
            text_embeddings = self.models["text_embedder"].encode(
                [t["content"] for t in texts], show_progress_bar=False
            )
                
        return {
            "texts": texts,
            "text_embeddings": np.array(text_embeddings),
            "max_page": max(t["page"] for t in texts) if texts else 0
        }
    
    def _extract_text(self, doc):
        texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            chunks = self.models["text_splitter"].split_text(page_text)
            for chunk in chunks:
                texts.append({
                    "content": chunk,
                    "page": page_num + 1,
                    "type": "text"
                })
        return texts

def generate_response(prompt, context, client):
    try:
        system_prompt = """You are a PDF Chat Bot. Provide:
1. Accurate information from the document
2. Well-structured responses with clear sections
3. Do not mention page numbers in the response
4. If document doesn't have the info, say 'This document does not contain information related to your query and give the answer in 2 lines only'"""
        
        response = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {prompt}"
            }],
            model=Config.GROQ_MODEL,
            temperature=0.33,
            max_tokens=Config.MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📖 PDF Chatbot")
    st.caption("Intelligent document analysis for text-based PDFs")
    
    # Initialize session state
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    models = load_models()
    processor = DocumentProcessor()
    
    with st.sidebar:
        st.header("Document Setup")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if st.button("Clear Session"):
            st.session_state.processed_data = None
            st.session_state.messages = []
            st.experimental_rerun()
    
    if uploaded_file and not st.session_state.processed_data:
        if not Config.GROQ_API_KEY:
            st.error("Missing GROQ_API_KEY in .env file")
            st.stop()
            
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
        
        with open(temp_pdf.name, "rb") as f:
            st.session_state.processed_data = processor.process_pdf(f.read())
        os.remove(temp_pdf.name)
        st.success("Document processed successfully!")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about the document..."):
        if not st.session_state.processed_data:
            st.error("Please upload a PDF first")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        data = st.session_state.processed_data
        response = ""
        text_refs = []
        
        with st.spinner("Analyzing request..."):
            # Handle page number validation
            page_match = re.search(r"page\s+(\d+)", prompt, re.IGNORECASE)
            if page_match:
                requested_page = int(page_match.group(1))
                if requested_page > data["max_page"]:
                    response = f"The document contains {data['max_page']} pages. Page {requested_page} does not exist."
            
            # Process text queries
            if not response:
                question_embed = models["text_embedder"].encode([prompt])
                text_scores = cosine_similarity(question_embed, data["text_embeddings"])[0]
                text_indices = np.argsort(text_scores)[-Config.TEXT_TOP_K:][::-1]
                context = "\n".join([f"Page {data['texts'][i]['page']}: {data['texts'][i]['content']}" 
                                   for i in text_indices])
                text_refs = [data["texts"][i] for i in text_indices]
                response = generate_response(prompt, context, models["groq_client"])
        
        with st.chat_message("assistant"):
            if response:
                # Clean up response formatting
                response = re.sub(r"I (apologize|don't have|can't provide)", "The document doesn't contain", response)
                response = re.sub(r"as (an|a) AI (language )?model", "based on this document", response)
                st.markdown(response)
                
                # Display text references if available
                if text_refs:
                    with st.expander("📄 Text References"):
                        for ref in text_refs:
                            st.write(f"**Page {ref['page']}**")
                            st.write(ref["content"])
                            st.divider()
            
            # Update message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "text_refs": text_refs
            })

if __name__ == "__main__":
    main()
