import streamlit as st
from app.youtube_transcript import get_transcript
from app.embedder import Embedder
from app.vector_store import VectorStore
from app.rag_pipeline import RAGPipeline

def main():
    st.title("YouTube RAG Assistant")
    
    # 1. Video Input
    video_url = st.text_input("Enter YouTube Video URL:")
    
    if video_url:
        with st.spinner("Processing video..."):
            # 2. Get Transcript
            transcript = get_transcript(video_url)
            
            # 3. Process and Store
            embedder = Embedder()
            chunks = embedder.process_text(transcript)
            
            vector_store = VectorStore()
            vector_store.initialize(chunks, embedder.embeddings)
            
            # 4. Initialize RAG
            rag = RAGPipeline(vector_store.retriever)
            
            st.success("Ready for questions!")
            
            # 5. Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Ask about the video"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.spinner("Thinking..."):
                    response = rag.answer_question(prompt)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()