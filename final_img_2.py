#    show image when it ask for image 
# but only one problem like low image Confidence level like 30.2%
# and also need the improve the text portion
import streamlit as st
import fitz  # PyMuPDF
import io
import re
import numpy as np
import os
import tempfile
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TEXT_EMBEDDER_MODEL = "all-mpnet-base-v2"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    GROQ_MODEL =    "llama3-8b-8192"# "llama3-70b-8192"
    IMAGE_TOP_K = 2
    TEXT_TOP_K = 5
    MAX_TOKENS = 500
    VISUAL_WEIGHT = 0.7
    TEXT_WEIGHT = 0.3
    MIN_IMAGE_CONFIDENCE = 0.25
    HISTORY_LIMIT = 4
    IMAGE_DISPLAY_WIDTH = 300
    IMAGE_KEYWORDS = r'\b(image|picture|pic|visual|graphic|photo|illustration)\b'

@st.cache_resource
def load_models():
    return {
        "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL),
        "clip_model": CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME),
        "clip_processor": CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME),
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
        
    def process_pdf(self, file_bytes) -> Dict[str, Any]:
        with st.spinner("📄 Analyzing document structure..."):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            texts, images = self._extract_content(doc)
            
            text_embeddings = self.models["text_embedder"].encode(
                [t["content"] for t in texts], show_progress_bar=False
            )
            
            image_embeddings = []
            for img in images:
                inputs = self.models["clip_processor"](
                    images=img["image"], return_tensors="pt"
                )
                features = self.models["clip_model"].get_image_features(**inputs)
                image_embeddings.append(features.detach().numpy())
                
        return {
            "texts": texts,
            "images": images,
            "text_embeddings": np.array(text_embeddings),
            "image_embeddings": np.concatenate(image_embeddings, axis=0) if image_embeddings else None,
            "has_images": len(images) > 0
        }
    
    def _extract_content(self, doc) -> tuple[List[Dict], List[Dict]]:
        texts = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Text extraction with enhanced cleaning
            page_text = re.sub(r'\s+', ' ', page.get_text("text")).strip()
            chunks = self.models["text_splitter"].split_text(page_text)
            for chunk in chunks:
                texts.append({
                    "content": chunk,
                    "page": page_num + 1,
                    "type": "text"
                })
            
            # Image extraction with metadata
            img_list = page.get_images(full=True)
            for img_info in img_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                try:
                    image = Image.open(io.BytesIO(base_image["image"]))
                    images.append({
                        "image": image,
                        "page": page_num + 1,
                        "type": "image",
                        "metadata": img_info[1:]
                    })
                except Exception as e:
                    continue
        
        return texts, images

class QueryProcessor:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.models = load_models()
        
    def build_context(self, history: List[Dict]) -> str:
        context = []
        for msg in history[-Config.HISTORY_LIMIT:]:
            if msg["role"] == "user":
                context.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                context.append(f"Assistant: {msg['content']}")
        return "\n".join(context)
    
    def process_image_query(self, query: str) -> tuple[str, List[Dict]]:
        if not self.data["has_images"]:
            return "📷 This document doesn't contain any images.", []

        search_query = extract_object_from_query(query) or query
        inputs = self.models["clip_processor"](
            text=[search_query], 
            return_tensors="pt", 
            padding=True
        )
        text_features = self.models["clip_model"].get_text_features(**inputs)
        image_scores = cosine_similarity(
            text_features.detach().numpy(), 
            self.data["image_embeddings"]
        )[0]
        
        filtered_images = []
        for idx, score in enumerate(image_scores):
            if score >= Config.MIN_IMAGE_CONFIDENCE:
                img = self.data["images"][idx]
                img["confidence"] = float(score)
                filtered_images.append(img)
                
        if not filtered_images:
            return f"🔍 No relevant images found for '{search_query}' (confidence < {Config.MIN_IMAGE_CONFIDENCE}).", []

        filtered_images.sort(key=lambda x: x["confidence"], reverse=True)
        return "🎨 Found these visual references:", filtered_images[:Config.IMAGE_TOP_K]
    
    def process_text_query(self, query: str, history: List[Dict]) -> tuple[str, List[Dict]]:
        question_embed = self.models["text_embedder"].encode([query])
        text_scores = cosine_similarity(question_embed, self.data["text_embeddings"])[0]
        text_indices = np.argsort(text_scores)[-Config.TEXT_TOP_K:][::-1]
        
        context = self.build_context(history)
        context += "\nRelevant text passages:\n" + "\n".join(
            [f"[Page {self.data['texts'][i]['page']}] {self.data['texts'][i]['content']}" 
             for i in text_indices]
        )
        
        response = generate_response(
            query, 
            context, 
            self.models["groq_client"],
            is_image_request=False
        )
        return response, [self.data['texts'][i] for i in text_indices]

def generate_response(prompt: str, context: str, client: Groq, is_image_request: bool) -> str:
    
    
    system_prompt = f"""You are a pdf Chat Bot. Provide:
1. Accurate information from the document
2. well structured responses with clear sections and required formate 
3. do nor mention the page number in the response

Context: {context}  
Question: {input}
"""
    
    
#     system_prompt = f"""🔍 You're a document analysis expert. Follow these rules:
# 1. Be concise and precise with information
# 2. Always cite sources using [Page X] notation
# 3. Acknowledge previous questions when relevant
# 4. Use emojis to make responses engaging
# 5. For image requests: {"Mention confidence levels" if is_image_request else ""}

# You are a document analysis expert. Follow these rules:
# 1. Combine text and images when requested
# 2. Acknowledge missing content explicitly
# 3. Structure response based on query type:

# **Image and Text rule** Give the text only but when it ask for image then follow these rule
# **Text-Only:** Formatted text with citations
# **Combined Query:** Text response + image grid
# **Image-Only:** Direct image display

# Context: {context}
# Question: {input}
# """
# 


    try:
        response = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuery: {prompt}"
            }],
            model=Config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=Config.MAX_TOKENS
        )
        return format_response(response.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

def format_response(text: str) -> str:
    text = re.sub(r'\[ ?Page (\d+) ?\]', r'📄 [Page \1]', text)
    text = re.sub(r'(^|\n)- ', r'\n• ', text)
    text = re.sub(r'(\d+\.) ', r'✨ \1 ', text)
    return text.strip()

def extract_object_from_query(query: str) -> str:
    match = re.search(r"show (?:me|the) (?:image|picture) of (.*)", query, re.IGNORECASE)
    return match.group(1).strip() if match else None

def main():
    st.set_page_config(page_title="Final_s5", layout="wide")
    st.title("📖 Document Visual Explorer")
    st.caption("Intelligent document analysis with multi-modal understanding")
    
    # Session state initialization
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("⚙️ Document Setup")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if st.button("🔄 Reset Session"):
            st.session_state.processed_data = None
            st.session_state.messages = []
            st.rerun()
    
    # File processing
    if uploaded_file and not st.session_state.processed_data:
        if not Config.GROQ_API_KEY:
            st.error("❌ Missing GROQ_API_KEY in .env file")
            st.stop()
            
        with st.status("🔍 Processing document...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(uploaded_file.getbuffer())
            
            with open(temp_pdf.name, "rb") as f:
                processor = DocumentProcessor()
                st.session_state.processed_data = processor.process_pdf(f.read())
            os.remove(temp_pdf.name)
            status.update(label="✅ Document processed!", state="complete")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("images"):
                cols = st.columns(min(3, len(msg["images"])))
                for idx, img in enumerate(msg["images"]):
                    caption = f"📄 Page {img['page']} (Confidence: {img.get('confidence', 0):.1%})"
                    cols[idx].image(img["image"], caption=caption, width=Config.IMAGE_DISPLAY_WIDTH)
            if msg.get("text_refs"):
                with st.expander(f"📚 {len(msg['text_refs'])} Text References"):
                    for ref in msg["text_refs"]:
                        st.markdown(f"**Page {ref['page']}**")
                        st.markdown(f"> {ref['content']}")
                        st.divider()
    
    # Handle new query
    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        data = st.session_state.processed_data
        processor = QueryProcessor(data)
        response = ""
        images = []
        text_refs = []
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("🔍 Analyzing query..."):
            # Determine query type
            is_visual = re.search(Config.IMAGE_KEYWORDS, prompt, re.IGNORECASE) is not None
            
            if is_visual:
                response, images = processor.process_image_query(prompt)
                if images:
                    response += f"\n\n🔍 Visual matches found with confidence ≥{Config.MIN_IMAGE_CONFIDENCE:.0%}"
            else:
                response, text_refs = processor.process_text_query(prompt, st.session_state.messages)
            
            # Add contextual emoji
            # if not response.startswith(("🎨", "🔍", "⚠️", "📄")):
            #     response = response
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(response)
            if images:
                cols = st.columns(min(3, len(images)))
                for idx, img in enumerate(images):
                    caption = f"📄 Page {img['page']} (Confidence: {img.get('confidence', 0):.1%})"
                    cols[idx].image(img["image"], caption=caption, width=Config.IMAGE_DISPLAY_WIDTH)
            if text_refs:
                with st.expander(f"📚 {len(text_refs)} Relevant Text Passages"):
                    for ref in text_refs:
                        st.markdown(f"**Page {ref['page']}**")
                        st.markdown(f"> {ref['content']}")
                        st.divider()
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "images": images,
            "text_refs": text_refs
        })

if __name__ == "__main__":
    main()
