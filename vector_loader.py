import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

if __name__ == "__main__":
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    folder_path = "./documents"
    docs = []
    
    if not os.path.exists('vectorstore'):
        os.makedirs('vectorstore')
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key="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"))
    vectorstore.save_local(DB_FAISS_PATH)





