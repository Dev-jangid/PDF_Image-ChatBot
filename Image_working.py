## Well working on single page images 


import streamlit as st
import fitz  # PyMuPDF
import io
import numpy as np
from PIL import Image
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TEXT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    GROQ_MODEL = "llama3-70b-8192"  # Current recommended model

# Initialize models and components
@st.cache_resource
def load_models():
    models = {
        "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL),
        "clip_model": CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME),
        "clip_processor": CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME),
        "text_splitter": RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
    }
    if Config.GROQ_API_KEY:
        models["groq_client"] = Groq(api_key=Config.GROQ_API_KEY)
    return models

class PDFProcessor:
    def __init__(self):
        self.models = load_models()
        
    def process_pdf(self, file_bytes):
        """Process uploaded PDF file"""
        with st.spinner("Extracting content..."):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            texts, images = self._extract_content(doc)
            
        with st.spinner("Generating embeddings..."):
            text_embeddings = self.models["text_embedder"].encode(
                [t["content"] for t in texts]
            )
            
            image_embeddings = []
            for img in images:
                inputs = self.models["clip_processor"](
                    images=img["image"], 
                    return_tensors="pt"
                )
                features = self.models["clip_model"].get_image_features(**inputs)
                image_embeddings.append(features.detach().numpy())
            
        return {
            "texts": texts,
            "images": images,
            "text_embeddings": np.array(text_embeddings),
            "image_embeddings": np.concatenate(image_embeddings, axis=0)
        }
    
    def _extract_content(self, doc):
        """Extract and split text and images from PDF"""
        texts = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract and split text
            page_text = page.get_text("text")
            chunks = self.models["text_splitter"].split_text(page_text)
            for chunk in chunks:
                texts.append({
                    "content": chunk,
                    "page": page_num + 1  # 1-based page numbering
                })
            
            # Extract images
            img_list = page.get_images(full=True)
            for img_info in img_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image = Image.open(io.BytesIO(base_image["image"]))
                images.append({
                    "image": image,
                    "page": page_num + 1,
                    "metadata": img_info[1:]
                })
        
        return texts, images

def generate_response(prompt, context, groq_client):
    """Generate answer using Groq API"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful PDF assistant. Answer based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {prompt}"
                }
            ],
            model=Config.GROQ_MODEL,
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(page_title="Image PDF Chatbot", layout="wide")
    st.title("📄 Image PDF Chatbot (Groq)")
    st.caption("Chat with documents containing text and images - Powered by Groq")
    
    # Initialize session state
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize models and processor
    models = load_models()
    processor = PDFProcessor()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        top_k = st.slider("Number of references", 1, 5, 3)
        st.markdown("---")
        
        if not Config.GROQ_API_KEY:
            st.error("Groq API key not found in .env file")
        else:
            st.success("Groq API key loaded successfully")
        
        st.caption(f"Using model: {Config.GROQ_MODEL}")
        st.caption("Note: Processing may take a minute for large PDFs")
    
    # Process PDF when uploaded
    if uploaded_file and not st.session_state.processed_data:
        if not Config.GROQ_API_KEY:
            st.error("Please configure GROQ_API_KEY in your .env file")
            return
            
        st.session_state.processed_data = processor.process_pdf(uploaded_file.read())
        st.success("PDF processed successfully!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("images"):
                cols = st.columns(len(message["images"]))
                for idx, img in enumerate(message["images"]):
                    cols[idx].image(img["image"], caption=f"Page {img['page']}")
    
    # Handle user input
    if prompt := st.chat_input("Ask about the PDF..."):
        if not st.session_state.processed_data:
            st.error("Please upload a PDF file first")
            return
        if not Config.GROQ_API_KEY:
            st.error("API key not configured")
            return
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.spinner("Analyzing document..."):
            data = st.session_state.processed_data
            
            # Text similarity search
            question_embedding = processor.models["text_embedder"].encode([prompt])
            text_scores = cosine_similarity(
                question_embedding, 
                data["text_embeddings"]
            )[0]
            text_indices = np.argsort(text_scores)[-top_k:][::-1]
            
            # Image similarity search
            text_inputs = processor.models["clip_processor"](
                text=[prompt], 
                return_tensors="pt", 
                padding=True
            )
            text_features = processor.models["clip_model"].get_text_features(**text_inputs)
            image_scores = cosine_similarity(
                text_features.detach().numpy(), 
                data["image_embeddings"]
            )[0]
            image_indices = np.argsort(image_scores)[-top_k:][::-1]
            
            # Prepare context
            context = "\n\n".join([
                f"Page {data['texts'][i]['page']}:\n{data['texts'][i]['content']}"
                for i in text_indices
            ])
            
            # Generate response
            answer = generate_response(prompt, context, models["groq_client"])
            
            # Get references
            text_refs = [data["texts"][i] for i in text_indices]
            image_refs = [data["images"][i] for i in image_indices]
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "text_refs": text_refs,
                "images": image_refs
            })
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # Show text references
            with st.expander("📄 Text References"):
                for ref in text_refs:
                    st.write(f"**Page {ref['page']}**")
                    st.write(ref["content"])
                    st.divider()
            
            # Show image references
            if image_refs:
                st.subheader("🖼️ Related Images")
                cols = st.columns(len(image_refs))
                for idx, img in enumerate(image_refs):
                    cols[idx].image(img["image"], caption=f"Page {img['page']}")

if __name__ == "__main__":
    main()
    

