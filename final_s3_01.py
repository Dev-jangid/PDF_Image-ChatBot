### does show the image in the Chat result without the releted section
# and has the perfect image size for the image 
### almost perfect text structured answer

## word on text and image_pdfs



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

# Load environment variables
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TEXT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CHUNK_SIZE = 1500  # Integer value
    CHUNK_OVERLAP = 300  # Integer value
    GROQ_MODEL = "llama3-70b-8192"
    IMAGE_TOP_K = 3  # Integer value
    TEXT_TOP_K = 5  # Integer value
    MAX_TOKENS = 500  # Integer value

@st.cache_resource
def load_models():
    return {
        "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL),
        "clip_model": CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME),
        "clip_processor": CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME),
        "text_splitter": RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,  # Corrected to use integer value
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
    
    def _extract_content(self, doc):
        texts = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Text extraction
            page_text = page.get_text("text")
            chunks = self.models["text_splitter"].split_text(page_text)
            for chunk in chunks:
                texts.append({
                    "content": chunk,
                    "page": page_num + 1,  # Integer page number
                    "type": "text"
                })
            
            # Image extraction
            img_list = page.get_images(full=True)
            for img_info in img_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image = Image.open(io.BytesIO(base_image["image"]))
                images.append({
                    "image": image,
                    "page": page_num + 1,  # Integer page number
                    "type": "image",
                    "metadata": img_info[1:]
                })
        
        return texts, images

def get_max_page(data):
    """Get maximum page number from processed data"""
    text_pages = [t["page"] for t in data["texts"]] if data["texts"] else []
    image_pages = [i["page"] for i in data["images"]] if data["images"] else []
    return max(text_pages + image_pages) if text_pages or image_pages else 0

def is_image_query(query):
    """Detect image-related requests using pattern matching"""
    image_patterns = [
        r"(show|display|view).*\b(image|picture|diagram|chart|figure|graph|photo)\b",
        r"(are there any|find|where is).*\b(images?|pictures?|diagrams?|charts?)\b",
        r"\b(page \d+).*(image|diagram|chart)\b",
        r"(related to|about).*\b(image|visual)\b"
    ]
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in image_patterns)

def extract_object_from_query(query):
    """Extract the specific object being requested from image query"""
    match = re.search(r"show (?:me|the) (?:image|picture) of (.*)", query, re.IGNORECASE)
    return match.group(1).strip() if match else None

def format_response(text):
    """Enhance response formatting"""
    text = re.sub(r'\n- ', '\n• ', text)
    text = re.sub(r'\[ ?Page (\d+) ?\]', r'[Page \1]', text)
    return text.strip()

def generate_response(prompt, context, client, is_image_request=False):
    try:
        system_prompt = """You are a pdf Chat Bot. Provide:
1. Accurate information from the document
2. well structured responses with clear sections and required formate 
2. do nor mention the page number in the response"""
# 2. Clear statements about image availability when requested"""
        # 2. Page citations [Page X] for all claims
        if is_image_request:
            system_prompt += "\n4. If images exist but none match the query, state this clearly"
        
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
    st.set_page_config(page_title="Final_s3_01", layout="wide")
    st.title("📖 Image-pdf Chatbot")
    st.caption("Intelligent document analysis with precise image handling")
    
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
            if message.get("images"):
                cols = st.columns(min(3, len(message["images"])))
                for idx, img in enumerate(message["images"]):
                    cols[idx].image(img["image"], caption=f"Page {img['page']}")
    
    if prompt := st.chat_input("Ask about the document..."):
        if not st.session_state.processed_data:
            st.error("Please upload a PDF first")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        data = st.session_state.processed_data
        response = ""
        images_to_show = []
        text_refs = []
        max_page = get_max_page(data)
        
        with st.spinner("Analyzing request..."):
            # Handle page number validation
            page_match = re.search(r"page\s+(\d+)", prompt, re.IGNORECASE)
            if page_match:
                requested_page = int(page_match.group(1))  # Ensure integer conversion
                if requested_page > max_page:
                    response = f"The document contains {max_page} pages. Page {requested_page} does not exist."
            
            # Process image queries
            is_img_query = is_image_query(prompt)
            requested_object = extract_object_from_query(prompt) if is_img_query else None
            
            if is_img_query and not page_match:
                if not data["has_images"]:
                    response = "This document does not contain any images."
                else:
                    search_query = requested_object if requested_object else prompt
                    inputs = models["clip_processor"](
                        text=[search_query], 
                        return_tensors="pt", 
                        padding=True
                    )
                    text_features = models["clip_model"].get_text_features(**inputs)
                    image_scores = cosine_similarity(
                        text_features.detach().numpy(), 
                        data["image_embeddings"]
                    )[0]
                    image_indices = np.argsort(image_scores)[-Config.IMAGE_TOP_K:][::-1]
                    images_to_show = [data["images"][i] for i in image_indices]
                    
                    if not images_to_show:
                        response = f"No matching images found for '{requested_object}' in this document." if requested_object else "No matching images found in this document."
                    else:
                        response = f"Here are the relevant images{f' containing {requested_object}' if requested_object else ''}:"
            
            # Process text queries
            if not response or (is_img_query and requested_object and not images_to_show):
                question_embed = models["text_embedder"].encode([prompt])
                text_scores = cosine_similarity(question_embed, data["text_embeddings"])[0]
                text_indices = np.argsort(text_scores)[-Config.TEXT_TOP_K:][::-1]
                context = "\n".join([f"Page {data['texts'][i]['page']}: {data['texts'][i]['content']}" 
                                   for i in text_indices])
                text_refs = [data["texts"][i] for i in text_indices]
                response = generate_response(
                    prompt, 
                    context, 
                    models["groq_client"],
                    is_image_request=is_img_query
                )
        
        with st.chat_message("assistant"):
            if response:
                # Clean up response formatting
                response = re.sub(r"I (apologize|don't have|can't provide)", "The document doesn't contain", response)
                response = re.sub(r"as (an|a) AI (language )?model", "based on this document", response)
                st.markdown(response)
                
                # Display images if available
                if images_to_show:
                    st.subheader("📷 Document Images")
                    cols = st.columns(min(3, len(images_to_show)))
                    for idx, img in enumerate(images_to_show):
                        cols[idx].image(img["image"], caption=f"Page {img['page']}")
                
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
                "images": images_to_show,
                "text_refs": text_refs
            })

if __name__ == "__main__":
    main()
