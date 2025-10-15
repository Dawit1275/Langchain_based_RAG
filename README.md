This is a Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. Here's what it does:

Key Features:
Document Processing:
Uploads and loads PDF files using PyPDFLoader
Splits documents into chunks using RecursiveCharacterTextSplitter
Creates embeddings using Azure OpenAI's text-embedding-ada-002 model
Vector Storage:
Uses Chroma vector database to store document embeddings
Enables semantic search across document content
Question Answering:
Uses Azure OpenAI's GPT-4o model for answering questions
Implements RetrievalQA chain to find relevant chunks and generate answers
Configurable number of documents to retrieve (k parameter)
User Interface:
Streamlit sidebar for API key input, file upload, and parameter configuration
Adjustable chunk size (50-1000 characters)
Question input field and answer display
Conversation history tracking
Workflow:
User enters Azure OpenAI API key
User uploads a PDF document
Document is processed into chunks and embedded
User asks questions about the document
System retrieves relevant chunks and generates answers
Maintains conversation history for reference
