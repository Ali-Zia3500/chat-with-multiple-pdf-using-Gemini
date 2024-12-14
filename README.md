# Chat with Multiple PDFs using GEMINI

# Overview:

This Streamlit web application allows users to interact with multiple PDF files by asking questions. It utilizes Google Generative AI (GEMINI) for natural language processing and vector storage with FAISS for efficient document retrieval.

# Features:

**PDF Upload:** 

Users can upload multiple PDF files.

**Text Extraction:**

Extracts text from uploaded PDF files.

**Text Chunking:**

Splits extracted text into smaller chunks for efficient processing.

**Vector Store:**

Saves text chunks into a vector database using FAISS for document retrieval.

**Q&A System:**

Provides conversational AI support to answer user queries from PDF documents.

**Error Handling:**

Provides clear error messages for troubleshooting.


# **Technologies Used**

Python

Streamlit

PyPDF2

LangChain

LLM: Google Generative AI (gemini-1.5-pro)

GoogleGenerativeAIEmbeddings

FAISS

# **Usage**

Upload multiple PDF files through the file uploader in the application.
Submit the uploaded PDFs for processing.
Ask questions about the content of the uploaded PDFs using the chat interface.

# **Setting Up the Google API Key**
https://aistudio.google.com/apikey

Login to Google AI Studio.

Open the link and log in with your Google account.

Access API Keys Section.

Once logged in, navigate to the API Keys tab.

Create an API Key.

Click on the "Generate API Key" button. This will create a new API key for your project.

Copy the API Key.

Save the API key securely. You'll need it to configure your project.

# **Install Dependencies**

pip install -r requirements.txt

# **Clone the repository:**

https://github.com/Ali-Zia3500/chat-with-multiple-pdf-using-Gemini.git

# **Run the Application**

streamlit run app.py

