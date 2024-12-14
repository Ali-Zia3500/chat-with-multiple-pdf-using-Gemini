import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google GenAI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key is not configured! Please check your .env file.")
    st.stop()
genai.configure(api_key=api_key)

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save the vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create the conversational chain for Q&A
def get_conservational_chain(retriever):
    prompt_template = """Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details.
    If the answer is not in the provided context, just say, "Answer is not available in the context."
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Load the vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()

        # Initialize the conversational chain
        chain = get_conservational_chain(retriever)

        # Pass the user question to the chain
        response = chain.run(query=user_question)
        
        # Display the response
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Load the vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()

        # Initialize the conversational chain
        chain = get_conservational_chain(retriever)

        # Pass the user question to the chain
        response = chain.run(query=user_question)
        
        # Display the response
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main application logic
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using GEMINI")

    user_question = st.text_input("Ask a question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Your PDFs and Click on Submit & Process", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"An error occurred while processing the PDFs: {e}")

if __name__ == "__main__":
    main()
