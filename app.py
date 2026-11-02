import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# Page configuration
st.set_page_config(
    page_title="Nikhil Singh | AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Load embedding model only once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Professional UI Design
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.main-title {
    text-align: center;
    color: #17365d;
    font-size: 35px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #52657a;
    font-size: 18px;
}
.student {
    text-align: center;
    color: #087e8b;
    font-size: 17px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Main heading
st.markdown(
    '<div class="main-title">🤖 Mini RAG-Based AI Chatbot</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Ask questions from your PDF documents</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="student">Developed by: Nikhil Singh</div>',
    unsafe_allow_html=True
)
st.divider()

# Sidebar
with st.sidebar:
    st.title("Project Details")
    st.write("**Student Name:** Nikhil Singh")
    st.write("**Project:** Mini RAG-Based AI Chatbot")
    st.write("**Technology Used:**")
    st.write("1. Python")
    st.write("2. Streamlit")
    st.write("3. FAISS")
    st.write("4. Sentence Transformers")
    st.write("5.Groq API -Llama")
    st.divider()
    st.info("Upload a PDF and ask questions based on its content.")

# PDF upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    reader = PdfReader(uploaded_file)

    # Store chunks with page numbers
    chunks = []
    page_numbers = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for i in range(0, len(page_text), 500):
            chunk = page_text[i:i+500].strip()

            if chunk:
                chunks.append(chunk)
                page_numbers.append(page_number)

    if not chunks:
        st.error("No readable text found in this PDF.")
        st.stop()

    st.success(f"PDF uploaded successfully! Total pages: {len(reader.pages)}")

    # Load embedding model
    with st.spinner("Loading AI embedding model..."):
        model = load_embedding_model()

    # Create FAISS index
    with st.spinner("Processing PDF..."):
        embeddings = model.encode(chunks)

        embeddings = np.array(embeddings).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

    st.success("PDF processing completed!")

    # Ask question
    question = st.text_input("Ask a question about your PDF")

    if question:

        with st.spinner("AI is generating your answer..."):

            query_embedding = model.encode([question]).astype("float32")

            k = min(3, len(chunks))

            distances, results = index.search(query_embedding, k=k)

            retrieved_chunks = []
            source_pages = set()

            for i in results[0]:
                if i >= 0:
                    retrieved_chunks.append(chunks[i])
                    source_pages.add(page_numbers[i])

            context = "\n".join(retrieved_chunks)

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "Answer only using the provided PDF context. Do not invent information. If the answer is not present, clearly say that you cannot find it in the document."
        },
        {
            "role": "user",
            "content": f"PDF Context:\n{context}\n\nQuestion: {question}"
        }
    ],
    temperature=0.2
)

        # Display answer
        st.subheader("Answer")
        st.write(response.choices[0].message.content)

        # Display source and page numbers
        st.subheader("📚 Sources")

        st.write("**PDF Name:**", uploaded_file.name)

        st.write(
            "**Information found on page(s):**",
            ", ".join(map(str, sorted(source_pages)))
        )

        for page in sorted(source_pages):
            st.write(f"✔ Page {page}")
