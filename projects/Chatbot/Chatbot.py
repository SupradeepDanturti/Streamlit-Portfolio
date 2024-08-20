import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os


st.title("Chatbot")

# Initialize chat history in session state if not already present
if "history" not in st.session_state:
    st.session_state.history = []

# Configure the model
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.1,
    convert_system_message_to_human=True
)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Vector Database Configuration
vector_directory = "./projects/Chatbot/db_dir/"  # Persist directory path
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(vector_directory):
    with st.spinner('🚀 Starting.... This might take a while'):
        # Data Pre-processing
        text_loader = DirectoryLoader("./projects/Chatbot/docs/", glob="./projects/Chatbot/docs/*.txt", loader_cls=TextLoader)
        text_documents = text_loader.load()
        print(f"Number of text documents loaded: {len(text_documents)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        text_context = "\n\n".join(str(p.page_content) for p in text_documents)
        print(f"Text Context Length: {len(text_context)}")
        texts = splitter.split_text(text_context)
        print(f"Number of texts after splitting: {len(texts)}")
        import glob

        files = glob.glob("./projects/Chatbot/docs/*.txt")
        print(f"Files found: {files}")

        data = texts
        print("Data Processing Complete")
        print(f"Data: {data[:5]}")
        vectordb = Chroma.from_texts(data, embeddings, persist_directory=vector_directory)
        vectordb.persist()

        print("Vector DB Creating Complete\n")

elif os.path.exists(vector_directory):
    vectordb = Chroma(persist_directory=vector_directory,
                      embedding_function=embeddings)

    print("Vector DB Loaded\n")

# Querying Model
query_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectordb.as_retriever()
)

# Displaying previous chat messages
for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Getting user input
prompt = st.chat_input("Say something")
if prompt:
    st.session_state.history.append({
        'role': 'user',
        'content': prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('💡Thinking'):
        response = query_chain({"query": prompt})

        st.session_state.history.append({
            'role': 'Assistant',
            'content': response['result']
        })

        with st.chat_message("Assistant"):
            st.markdown(response['result'])
