import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
# from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path

# set_debug(True)

os.environ["GOOGLE_API_KEY"] = "AIzaSyBpgW7nF6ShlFVo9kxbMtpDPVYM_-Po6e4"

st.title("Gullus Jam !")

with st.sidebar:
    st.title("Gullus Jam !")
    username=st.text_input("Enter your home name")

if not username :
    st.warning("Please enter your name your husband calls you.")
    st.stop()

if  username.strip().lower()!="gullu":
    st.warning("Enter your name that sarts with G to proceed.")
    st.stop()


embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key="AIzaSyBpgW7nF6ShlFVo9kxbMtpDPVYM_-Po6e4"
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

BASE_DIR = Path(__file__).parent
file_path = BASE_DIR /"gullus.txt"

print("filepath",file_path)
documents = TextLoader(file_path).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(documents)

vector_store=Chroma.from_documents(chunks, embedding_model)

retriever = vector_store.as_retriever()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for answering questions.
    Use the provided context to respond.If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
         """),
        ("human", "{input}")
    ]
)


qa_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever, qa_chain)

question = st.text_input("Enter the query: ")

response = rag_chain.invoke({"input":question})
st.write(response['answer'])
