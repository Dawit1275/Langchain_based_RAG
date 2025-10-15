import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

# Using updated Chroma import
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Function to get embeddings (avoids global variable)
def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_type="azure",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model="text-embedding-ada-002"
    )

# Load PDF
def load_document(file):
    print(f"Loading {file}...")
    loader = PyPDFLoader(file)
    data = loader.load()  # one document per page
    return data

# Chunk text
def chunk_data(data, chunk_size=256,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings and vector store
def create_embeddings_chroma(chunks):
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_question(question, vector_store,k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT"),
        temperature=0,
        model="gpt-4o",
    )
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa_chain.run(question)
    return response

def clear_history():
    if "history" in st.session_state:
        del st.session_state['history']

st.image('img.jpg')
st.subheader("LLM question answering Application")
with st.sidebar:
    api_key = st.text_input("Enter your Azure OpenAI API Key:", type="password")
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    uploaded_file = st.file_uploader("Upload a file", type=["pdf"])
    chunk_size = st.number_input("Chunk Size", min_value=50, max_value=1000, value=256,on_change=clear_history)
    k = st.number_input("Number of documents to retrieve (k)", min_value=1, max_value=10, value=3,on_change=clear_history)
    add_data = st.button("Add & Process Document",on_click=clear_history)

    if uploaded_file and add_data:
        with st.spinner("Processing document..."):
            bytes_data = uploaded_file.read()
            file_name=os.path.join('./', uploaded_file.name)
            with open(file_name, "wb") as f:
                f.write(bytes_data)

            data = load_document(file_name)
            chunks = chunk_data(data, chunk_size=chunk_size)
            st.write(f"chunk size:{chunk_size},Total chunks created: {len(chunks)}")
            vector_store = create_embeddings_chroma(chunks)
            st.session_state.vs=vector_store
            st.success("Document processed and embeddings created.")

q = st.text_input("Enter your question:")
answer = ""  # <-- Add this line to avoid NameError
if q:
    if 'vs' in st.session_state:
        vector_store = st.session_state.vs
        st.write(f"k: {k}")
        answer = ask_question(q, vector_store,k) 
        st.text_area("Answer", value=answer, height=200)
        st.divider()
        if "history" not in st.session_state:
            st.session_state.history = []
        value = f"Q: {q} \n A: {answer}"
        st.session_state.history = f"{value}\n{"_"*80}\n {st.session_state.history}"
        h = st.session_state.history
        st.text_area(label ="Conversation History", value=h, key ="history", height=300)



