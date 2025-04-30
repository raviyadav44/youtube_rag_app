#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YouTube RAG Assistant - Enhanced Version"""

# ===== Initial Configuration =====
import os
import shutil
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix event loop before any other imports
import asyncio
import nest_asyncio
nest_asyncio.apply()

# GPU Configuration
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch._C._disable_torch_function_override = True
if hasattr(torch._classes, '__path__'):
    del torch._classes.__path__

# ===== Main Imports =====
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path
import hashlib
from typing import List, Tuple
import textwrap

# ===== Enhanced Configuration =====
# Paths
BASE_DIR = Path(__file__).parent
USER_DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = USER_DATA_DIR / "chroma_db"

# Model Choices
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "google/flan-t5-large"

# Chunking Parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 200  # Increased for better context continuity

# Retrieval Parameters
TOP_K = 8  # Increased from 5 for more context

# Ensure directories exist
os.makedirs(CHROMA_DIR, exist_ok=True)

# App constants
DEBUG_EXPANDER = True
MAX_CHUNK_PREVIEW = 500

# ===== Enhanced Core Components =====
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YouTube RAG Assistant - Reliable Version"""

# ===== Initial Configuration =====
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import nest_asyncio
nest_asyncio.apply()

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Main Imports =====
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path
import hashlib
from typing import List, Tuple

# ===== Configuration =====
BASE_DIR = Path(__file__).parent
USER_DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = USER_DATA_DIR / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "google/flan-t5-large"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K = 4

os.makedirs(CHROMA_DIR, exist_ok=True)

# ===== Core Components =====
class OpenSourceLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            temperature=0.3,  # More focused answers
            do_sample=False  # More deterministic
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

class EnhancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
        self.qa_prompt = """Answer the question based on the context below.
Keep your answer concise but complete, using proper grammar and punctuation.

Context: {context}

Question: {question}

Answer:"""
    
    def answer_question(self, query: str) -> Tuple[str, List[Document]]:
        try:
            docs = self.retriever.get_relevant_documents(query)
            context = "\n".join(doc.page_content for doc in docs)
            
            prompt = self.qa_prompt.format(
                context=context,
                question=query
            )
            
            answer = self.llm(prompt)
            
            # Simple cleanup
            answer = answer.strip()
            if answer and answer[-1] not in {'.', '!', '?'}:
                answer += '.'
            if answer:
                answer = answer[0].upper() + answer[1:]
                
            return answer, docs
        except Exception as e:
            return f"Error: {str(e)}", []

class Embedder:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_text(self, text: str, video_url: str) -> List[Document]:
        chunks = self.splitter.create_documents([text])
        namespace = hashlib.md5(video_url.encode()).hexdigest()[:8]
        
        for chunk in chunks:
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            
            chunk.metadata.update({
                "video_id": namespace,
                "video_url": video_url,
                "source": video_url,
                "chunk_id": f"{namespace}_{len(chunk.page_content)}"
            })
        
        return chunks

class VectorStore:
    def __init__(self):
        self.persist_dir = CHROMA_DIR
    
    def _get_video_namespace(self, video_url: str) -> str:
        return hashlib.md5(video_url.encode()).hexdigest()[:8]
    
    def initialize(self, chunks, embeddings, video_url: str):
        namespace = self._get_video_namespace(video_url)
        
        try:
            existing = Chroma(
                collection_name=f"video_{namespace}",
                persist_directory=str(self.persist_dir),
                embedding_function=embeddings
            )
            existing.delete_collection()
        except:
            pass
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"video_{namespace}",
            persist_directory=str(self.persist_dir)
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": TOP_K,
                "filter": {"video_id": namespace}
            }
        )
        return self.retriever


# ===== Utility Functions =====
def extract_video_id(url: str) -> str:
    """Extract ID from various YouTube URL formats"""
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:
        query = urlparse(url).query
        return parse_qs(query).get("v", [None])[0]
    return url  # Assume raw ID

def get_transcript(video_input: str, languages=["en"]):
    try:
        video_id = extract_video_id(video_input)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        print(f"Transcripts disabled for video: {video_id}")
        return None
    except Exception as e:
        print(f"Transcript error: {str(e)}")
        return None

# ===== Application Functions =====
def initialize_session_state():
    """Initialize all session state variables"""
    st.session_state.setdefault("debug", {
        "current_video": None,
        "transcript_loaded": False,
        "chunk_count": 0,
        "vector_store_ready": False,
        "llm_loaded": False,
        "errors": []
    })
    st.session_state.setdefault("messages", [])

def clear_previous_video():
    """Reset state when new video is loaded"""
    st.session_state.debug = {
        "current_video": None,
        "transcript_loaded": False,
        "chunk_count": 0,
        "vector_store_ready": False,
        "llm_loaded": False,
        "errors": []
    }
    st.session_state.messages = []

def show_debug_panel(chunks=None):
    """Display debug information in expandable panel"""
    if not DEBUG_EXPANDER:
        return
        
    with st.expander("🔧 Debug Panel", expanded=False):
        st.subheader("Pipeline State")
        st.json(st.session_state.debug)
        
        if chunks and len(chunks) > 0:
            st.divider()
            st.subheader("Sample Chunk")
            st.code(chunks[0].page_content[:MAX_CHUNK_PREVIEW] + "...", language="text")
            st.write("Chunk metadata:", chunks[0].metadata)

def process_video(video_url):
    """Handle the full RAG pipeline for a YouTube video"""
    with st.status("🎥 Processing video...", expanded=True) as status:
        try:
            # Track current video
            st.session_state.debug["current_video"] = video_url
            
            # === STAGE 1: Transcript ===
            status.update(label="📜 Fetching transcript...")
            transcript = get_transcript(video_url)
            
            if not transcript:
                st.error("❌ No transcript available for this video")
                return None
                
            st.session_state.debug["transcript_loaded"] = True

            # === STAGE 2: Chunking ===
            status.update(label="✂️ Chunking text...")
            embedder = Embedder()
            chunks = embedder.process_text(transcript, video_url)  # Pass video_url for metadata
            st.session_state.debug["chunk_count"] = len(chunks)

            # === STAGE 3: Vector Store ===
            status.update(label="🗄️ Creating vector store...")
            vector_store = VectorStore()
            vector_store.initialize(chunks, embedder.embeddings, video_url)  # Pass video_url
            st.session_state.debug["vector_store_ready"] = True

            # === STAGE 4: LLM ===
            status.update(label="🧠 Loading language model...")
            llm = OpenSourceLLM().llm
            st.session_state.debug["llm_loaded"] = True

            status.update(label="✅ Ready for questions!", state="complete")
            return EnhancedRAGPipeline(vector_store.retriever, llm)
            
        except Exception as e:
            st.session_state.debug["errors"].append(str(e))
            status.update(label="❌ Processing failed", state="error")
            st.error(f"Pipeline error: {str(e)}")
            return None

def chat_interface(rag):
    """Enhanced chat interface with context display"""
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show context if available
            if message["role"] == "assistant" and "context" in message:
                with st.expander("🔍 See sources used"):
                    for i, doc in enumerate(message["context"]):
                        st.write(f"#### Source Chunk {i+1}")
                        st.code(textwrap.fill(doc.page_content, width=80))
                        st.caption(f"Source: {doc.metadata.get('source', 'video')}")
    
    # Handle new question
    if prompt := st.chat_input("Ask about the video"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("💭 Analyzing video content..."):
            try:
                answer, docs = rag.answer_question(prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "context": docs  # Store context for display
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to generate answer: {str(e)}")
                st.session_state.debug["errors"].append(str(e))


# ===== Main Application =====
def main():
    """Main application function"""
    st.title("YouTube RAG Assistant 🎥🔍")
    
    # Initialize session state
    if "debug" not in st.session_state:
        initialize_session_state()
    
    # Video input
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        key="video_input"
    )
    
    # Check if new video was entered
    if video_url and video_url != st.session_state.debug.get("current_video"):
        clear_previous_video()
    
    # Process video if URL exists and not already processed
    if video_url and not st.session_state.debug.get("llm_loaded"):
        if rag := process_video(video_url):
            st.session_state.rag = rag
    
    # Chat interface if ready
    if st.session_state.get("rag"):
        chat_interface(st.session_state.rag)
    
    # Debug panel
    show_debug_panel()

if __name__ == "__main__":
    main()
