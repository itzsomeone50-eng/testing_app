import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Initialize Groq LLM
llm = ChatGroq(
    api_key = "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",
    model_name="llama-3.3-70b-versatile",
    temperature=0.0)
# =========================================
# Streamlit App UI
# =========================================
st.set_page_config(page_title="RAG PDF QA", page_icon="📄", layout="centered")
st.title("📄 RAG-based PDF Question Answering Assistant")

# Initialize session state variables (to avoid re-running)
if "summary" not in st.session_state:
    st.session_state.summary = None
if "recommended_questions" not in st.session_state:
    st.session_state.recommended_questions = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# =========================================
# 1. File Upload
# =========================================
uploaded_file = st.file_uploader("📤 Upload your PDF document", type="pdf")
# Instruction dropdown with predefined styles
instruction_options = [
    "Default (No special formatting)",
    "Answer in bullet points",
    "Keep it short and concise",
    "Explain like I'm 5",
    "Provide detailed explanation",
    "Custom"
]

selected_instruction = st.selectbox("✍️ Choose instruction style for AI:", instruction_options)

# If 'Custom' is selected, show a text box for custom instructions
if selected_instruction == "Custom":
    instructions = st.text_area(
        "✍️ Enter your custom instructions:",
        placeholder="E.g., Use formal tone, explain step-by-step..."
    )
else:
    # Map predefined instructions to actual text prompts
    instruction_map = {
        "Default (No special formatting)": "",
        "Answer in bullet points": "Provide the answer in clear bullet points.",
        "Keep it short and concise": "Give a very short and concise answer (2-3 lines).",
        "Explain like I'm 5": "Explain the answer in very simple terms like explaining to a 5-year-old.",
        "Provide detailed explanation": "Give a detailed explanation with proper reasoning."
    }
    instructions = instruction_map[selected_instruction]

# Question input
query = st.text_input("🔎 Enter your question:", placeholder="E.g., Summarize this report in 3 bullet points")

# Answer output area
answer_placeholder = st.empty()

# =========================================
# Backend Processing (RAG Flow)
# =========================================
if uploaded_file is not None:
    # ✅ Process file ONLY when a new file is uploaded
    if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name  # Track current file

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load and process the PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        # Create embeddings and vector DB (FAISS)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        # Save retriever to session
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever

        # ===============================
        # Generate About File (Summary) & Recommended Questions ONCE
        # ===============================
        summary_prompt = PromptTemplate(
            input_variables=["context"],
            template="""
            Summarize the following document in 2 - 3 lines 
            explaining its core content and purpose keeping concise:
            
            Context:
            {context}

            Summary:
            """
        )

        # Use first few chunks for summary context
        initial_context = " ".join([doc.page_content for doc in split_docs[:3]])
        summary_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": summary_prompt}
        )
        st.session_state.summary = summary_chain.invoke({"query": "Summarize the document."})["result"]

        # Recommended Questions
        questions_prompt = PromptTemplate(
            input_variables=["context"],
            template="""
            Based on the following document content, suggest 3-4 relevant 
            and helpful questions that shpuld be aslked by me also make them max of 1.5 lines each

            Context:
            {context}

            Suggested Questions:
            """
        )
        questions_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": questions_prompt}
        )
        st.session_state.recommended_questions = questions_chain.invoke({"query": "Suggest relevant questions."})["result"]

    # ===============================
    # Display Summary & Recommended Questions (Persisted in Session)
    # ===============================
    st.subheader("📑 About the Uploaded File")
    st.write(st.session_state.summary)

    st.subheader("💡 Recommended Questions to Ask")
    st.text_area("Recommended Questions:", value=st.session_state.recommended_questions, height=320)

    # ===============================
    # Answer User Query (Runs without regenerating summary/questions)
    # ===============================
    if query:
        rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=f"""
            You are an intelligent assistant. Your task is to ONLY answer questions using 
            the information from the provided context (uploaded document).

            {instructions}

            Rules:
            - If the answer is NOT present in the context, reply: 
            "❌ I cannot answer this as it is not found in the uploaded document."
            - Do NOT use external knowledge or make assumptions.

            Context:
            {{context}}

            Question:
            {{question}}

            Answer:
            """
        )
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": rag_prompt},
            return_source_documents=True
        )

        with st.spinner("🤖 Reading your document and generating answer..."):
            response = rag_chain.invoke({"query": query})
            if len(response["source_documents"]) == 0:
                st.warning("⚠️ No relevant information found in the document.")
            else:
                st.markdown(f"### ✅ Answer:\n{response['result']}")

else:
    st.info("📌 Upload a PDF and enter a question to get started.")

