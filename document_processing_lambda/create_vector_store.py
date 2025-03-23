"""
Functions for ingesting txt, csv files into Pinecone vectorstore
Currently meant to be run locally
"""

from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader
from pinecone import Pinecone
import os
from os import getenv
from dotenv import load_dotenv
from typing import Optional
from uuid import uuid4



def upload_to_pinecone(filename: str, chunk_size:int = 200, chunk_overlap:int = 50, index_name:str = 'gaim-bot-index') -> PineconeVectorStore:
    """process document .txt or .csv into vector store as a langchain document instance

    Args:
        filename: str
        chunk_size:int=200
        chunk_overlap:int=50
        index_name:str = 'gaim-bot-index'

    Returns:
        PineconeVectorStore
    """

    # Extract file ext to determine file type
    _, ext = filename.split('.') 
    if ext == 'csv':
        loader = CSVLoader(filename)
    elif ext == 'txt':
        loader = TextLoader(filename)
    else:
        raise TypeError('incorrect file type')
    documents = loader.load()

    # Split documents for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(documents)

    # Load Pinecone Index, must already be created
    pc = Pinecone(api_key=getenv('PINECONE_API_KEY'))
    pc_index = pc.Index(index_name)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
    vector_store = PineconeVectorStore(index=pc_index, embedding=embeddings)

    # Add document to vector_store with a uuid
    embeddings = OpenAIEmbeddings()
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=split_documents, ids=uuids)

    return vector_store


if __name__ == '__main__':
    load_dotenv()
    openai_api_key = getenv('OPENAI_API_KEY')
    folder_path = "" # ADD PATH TO FILES  

    for filename in os.listdir(folder_path):
        print('Processing file: ', filename)
        if filename == "locations.csv":
            continue
        file_path = folder_path + '/' + filename
        print('Verifying file path: ', file_path)
        if os.path.isfile(file_path):  
            try:
                upload_to_pinecone(file_path)
            except Exception as e:
                print(f"Error reading file: {filename} with error: {e}")



