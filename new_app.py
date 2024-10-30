import json
import os
import sys
import boto3
import streamlit as st
from PIL import Image
import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Include pytesseract for OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Update with your Tesseract path

def save_uploadedfile(uploadedfile):
    """Save uploaded files to a directory named 'file'."""
    if not os.path.exists("file"):
        os.makedirs("file")
    file_path = os.path.join("file", uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path


def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


## We will be using Titan Embeddings Model To generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
# Calling the bedrock embedding
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def data_ingestion(uploaded_files):
    # Initialize a list to hold all the documents
    documents = []

    for uploaded_file in uploaded_files:
        # Save the uploaded file locally to 'file' directory
        file_path = save_uploadedfile(uploaded_file)

        if uploaded_file.name.endswith(".pdf"):
            # Use PyPDF loader for PDF files
            pdf_loader = PyPDFLoader(file_path)
            pdf_documents = pdf_loader.load()
            documents.extend(pdf_documents)

        elif uploaded_file.name.endswith(".jpg") or uploaded_file.name.endswith(".jpeg") or uploaded_file.name.endswith(".png"):
            # Process image files for OCR
            text = extract_text_from_image(file_path)
            document = Document(page_content=text, metadata={"source": uploaded_file.name})
            documents.append(document)

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    return docs


## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")


def get_llama3_llm():
    ## Create the Anthropic Model
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock,
                  model_kwargs={'max_gen_len': 512})
    
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 250 words with detailed explanations. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF & Images")

    st.header("Chat with PDF & Images using AWS Bedrock 💁")

    user_question = st.text_input("Ask a Question from the PDF Files or Images")

    with st.sidebar:
        st.title("File Upload and Vector Store:")

        # Upload PDF or Image files in the sidebar, just above "Vectors Update"
        uploaded_files = st.file_uploader("Upload PDF or Image files", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if st.button("Vectors Update"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    docs = data_ingestion(uploaded_files)
                    get_vector_store(docs)
                    st.success("Vector store updated.")
            else:
                st.warning("Please upload files to update vectors.")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            if user_question:
                st.write(get_response_llm(llm, faiss_index, user_question))
            else:
                st.warning("Please enter a question to get a response.")
            st.success("Done")


if _name_ == "_main_":
    main()
