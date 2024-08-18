import os
import streamlit as st
import pickle
import time
import langchain
from langchain_openai import AzureChatOpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader, DirectoryLoader
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from langchain.chains.retrieval_qa.base import VectorDBQA
from dotenv import load_dotenv



langchain.debug = True
load_dotenv()  # take environment variables from .env (especially openai api key)



TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "files")



def azure_llm():
    llm = AzureChatOpenAI(
        azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model = "gpt-4o",
    )
    return llm

def azure_embeddings():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT'),
        model="text-embedding-3-large",
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
    )
    return embeddings



st.title("The one - All in one research tool")
st.sidebar.title("Sources of your research")


question = st.text_input("Question: ")
markdown = st.empty()


# Handle pdf
add_files = st.sidebar.file_uploader("Add PDF")

pdf_upload_path = 'data/files'

if add_files:
    with open(os.path.join(TMP_DIR, add_files.name), 'wb') as f:
        f.write(add_files.getbuffer())


# Handle urls
urls = []
for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_clicked = st.sidebar.button("Process Sources")




# RAG start

def load_docs():
    docs = []
    pdf_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    docs.extend(pdf_loader.load())

    url_loader = UnstructuredURLLoader(urls=urls)
    docs.extend(url_loader.load())

    return docs

if process_clicked:
    print(load_docs())
    markdown.write("Documents Loaded Successfully")

    # Split
#
persist_directory = 'db'
if question:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(load_docs())

    # Embed


    vectorstore = Chroma.from_documents(documents=splits, embedding=azure_embeddings(), persist_directory=persist_directory)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'lambda_mult': 0.25})



    #### RETRIEVAL and GENERATION ####

    # Prompt


    # HyDE document generation
    hyde_template = """Generate a hypothetical document from the question that you will recieve, provide generic information to the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(hyde_template)

    generate_docs_for_retrieval = (
        prompt_hyde | azure_llm() | StrOutputParser()
    )

    # Run


    # Retrieve
    retrieval_chain = generate_docs_for_retrieval | retriever
    retireved_docs = retrieval_chain.invoke({"question":question})

    #RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | azure_llm()
        | StrOutputParser()
    )

    result = final_rag_chain.invoke({"context": retireved_docs, "question": question})
    st.header("Answer")
    st.write("Question:", question)
    st.write(result)
    print(final_rag_chain.invoke({"context": retireved_docs, "question": question}))


