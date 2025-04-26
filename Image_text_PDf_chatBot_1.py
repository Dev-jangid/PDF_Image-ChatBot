## better for image+text  
#  always show the image along with the text in releted images box
#   but has the errorr does not finds the releted image

import streamlit as st
import fitz  # PyMuPDF
import io
import re
import numpy as np
import os
import tempfile
import pytesseract
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

# Load environment variables
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TEXT_EMBEDDER_MODEL = "all-mpnet-base-v2"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    GROQ_MODEL = "llama3-70b-8192"
    IMAGE_TOP_K = 3
    TEXT_TOP_K = 5
    MAX_TOKENS = 1200
    MIN_IMAGE_CONFIDENCE = 0.25

@st.cache_resource
def load_models():
    try:
        pytesseract.get_tesseract_version()
    except EnvironmentError:
        st.error("Tesseract OCR not installed. Required for image text analysis.")
        st.stop()
        
    return {
        "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL),
        "clip_model": CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME),
        "clip_processor": CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME),
        "text_splitter": RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". "]
        ),
        "groq_client": Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    }

class DocumentProcessor:
    def __init__(self):
        self.models = load_models()
        self.image_cache = {}

    def process_pdf(self, file_bytes):
        """Process PDF with enhanced image-text correlation"""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts, images = self._extract_content(doc)
        
        return {
            "texts": texts,
            "images": self._process_images_with_ocr(images),
            "text_embeddings": self._embed_texts(texts),
            "image_embeddings": self._embed_images(images)
        }

    def _extract_content(self, doc):
        """Extract text and images with page correlation"""
        texts = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            texts += self._chunk_text(page_text, page_num)
            images += self._extract_page_images(page, doc, page_num)
        
        return texts, images

    def _chunk_text(self, text, page_num):
        """Split text with page-aware chunking"""
        if not text.strip():
            return []
            
        chunks = self.models["text_splitter"].split_text(text)
        return [{
            "content": chunk,
            "page": page_num + 1,
            "hash": hashlib.md5(chunk.encode()).hexdigest()
        } for chunk in chunks]

    def _extract_page_images(self, page, doc, page_num):
        """Extract images with deduplication"""
        images = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in self.image_cache:
                images.append(self.image_cache[xref])
                continue
                
            try:
                base_image = doc.extract_image(xref)
                image = Image.open(io.BytesIO(base_image["image"]))
                img_data = {
                    "image": image,
                    "page": page_num + 1,
                    "metadata": img_info[1:],
                    "hash": hashlib.md5(base_image["image"]).hexdigest()
                }
                self.image_cache[xref] = img_data
                images.append(img_data)
            except Exception as e:
                st.warning(f"Skipped image: {str(e)}")
        return images

    def _process_images_with_ocr(self, images):
        """Add OCR text to image metadata"""
        for img in images:
            try:
                img["ocr_text"] = pytesseract.image_to_string(img["image"])[:200]
            except:
                img["ocr_text"] = ""
        return images

    def _embed_texts(self, texts):
        return self.models["text_embedder"].encode(
            [t["content"] for t in texts],
            convert_to_tensor=True
        )

    def _embed_images(self, images):
        if not images:
            return np.array([])
            
        return np.concatenate([
            self.models["clip_model"].get_image_features(
                **self.models["clip_processor"](images=img["image"], return_tensors="pt")
            ).detach().numpy()
            for img in images
        ], axis=0)

class QueryAnalyzer:
    def __init__(self, models):
        self.models = models
        
    def analyze(self, query, data):
        """Enhanced analysis with text-image correlation"""
        text_results = self._analyze_text(query, data)
        image_results = self._analyze_images(query, data, text_results["pages"])
        
        return {
            "text": text_results,
            "images": image_results
        }

    def _analyze_text(self, query, data):
        """Text analysis with page correlation"""
        query_embed = self.models["text_embedder"].encode(query, convert_to_tensor=True)
        scores = cosine_similarity(query_embed.unsqueeze(0), data["text_embeddings"])[0]
        top_indices = np.argsort(scores)[-Config.TEXT_TOP_K:][::-1]
        
        return {
            "context": "\n".join(
                f"Page {data['texts'][i]['page']}: {data['texts'][i]['content']}" 
                for i in top_indices
            ),
            "references": [data['texts'][i] for i in top_indices],
            "pages": {data['texts'][i]['page'] for i in top_indices}
        }

    def _analyze_images(self, query, data, text_pages):
        """Image analysis considering both visual and textual relevance"""
        if not data["images"]:
            return {"context": "", "references": []}
            
        # Visual similarity
        text_inputs = self.models["clip_processor"](text=[query], return_tensors="pt")
        text_features = self.models["clip_model"].get_text_features(**text_inputs)
        visual_scores = cosine_similarity(
            text_features.detach().numpy(), 
            data["image_embeddings"]
        )[0]
        
        # Textual relevance
        text_scores = [
            1 if (img["page"] in text_pages or 
                  re.search(query, img["ocr_text"], re.IGNORECASE)) 
            else 0 
            for img in data["images"]
        ]
        
        # Combined scores
        combined_scores = [
            0.7 * visual + 0.3 * text 
            for visual, text in zip(visual_scores, text_scores)
        ]
        
        top_indices = np.argsort(combined_scores)[-Config.IMAGE_TOP_K:][::-1]
        valid_indices = [i for i in top_indices if combined_scores[i] > Config.MIN_IMAGE_CONFIDENCE]
        
        return {
            "context": "\n".join(
                f"Page {data['images'][i]['page']} image contains: {data['images'][i]['ocr_text']}"
                for i in valid_indices
            ),
            "references": [data['images'][i] for i in valid_indices]
        }

class ResponseGenerator:
    def __init__(self, client):
        self.client = client
        
    def generate(self, query, text_context, image_context, has_images):
        """Generate response with accurate image-text correlation"""
        system_prompt = f"""You are a document analysis expert. Follow these rules:
1. Acknowledge text mentions even without images
2. Clearly state when images are referenced but unavailable
3. Differentiate between text references and actual images
4. Never hallucinate images that don't exist

Text Context: {text_context}
Image Context: {image_context if has_images else 'No relevant images found'}"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": query
                }],
                model=Config.GROQ_MODEL,
                temperature=0.3,
                max_tokens=Config.MAX_TOKENS
            )
            return self._format_response(response.choices[0].message.content)
        except Exception as e:
            return f"Error: {str(e)}"

    def _format_response(self, text):
        """Clean up response formatting"""
        text = re.sub(r'\n- ', '\n• ', text)
        text = re.sub(r'\[ ?Page (\d+) ?\]', r'[Page \1]', text)
        return text.strip()

def main():
    st.set_page_config(page_title="Image_text_PDf_chatBot_1", layout="wide")
    st.title("📚 Image_text_PDf_chatBot_1")
    st.caption("Accurate text-image correlation with contextual understanding")
    
    # Initialize components
    models = load_models()
    processor = DocumentProcessor()
    analyzer = QueryAnalyzer(models)
    responder = ResponseGenerator(models["groq_client"])
    
    # Session state management
    if "doc_data" not in st.session_state:
        st.session_state.doc_data = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar controls
    with st.sidebar:
        st.header("Document Control")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if st.button("Clear Session"):
            st.session_state.clear()
            st.rerun()

    # File processing
    if uploaded_file and not st.session_state.doc_data:
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
        
        with open(temp_pdf.name, "rb") as f:
            st.session_state.doc_data = processor.process_pdf(f.read())
        os.remove(temp_pdf.name)
        st.success("Document analysis complete!")

    # Chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("images"):
                with st.expander("Related Images"):
                    cols = st.columns(min(3, len(msg["images"])))
                    for idx, img in enumerate(msg["images"]):
                        cols[idx].image(img["image"], caption=f"Page {img['page']}")

    # Query handling
    if prompt := st.chat_input("Ask about the document..."):
        if not st.session_state.doc_data:
            st.error("Please upload a document first")
            return
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process query
        analysis = analyzer.analyze(prompt, st.session_state.doc_data)
        response = responder.generate(
            prompt,
            analysis["text"]["context"],
            analysis["images"]["context"],
            bool(analysis["images"]["references"])
        )
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "images": analysis["images"]["references"]
        })
        
        # Rerun to update display
        st.rerun()

if __name__ == "__main__":
    main()
