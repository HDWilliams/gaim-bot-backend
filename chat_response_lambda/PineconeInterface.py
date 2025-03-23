"""
Class for interacting with pinecone vectorstore
"""
import logging
from os import getenv
from typing import List
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, exceptions

class PineconeInterface:
    def __init__(self, index_name:str) -> None:
        self.vector_store = self.get_pinecone_vector_store(index_name)

    def get_pinecone_vector_store(self, index_name:str) -> PineconeVectorStore:
        """Create object for interacting with specific pinecone vector store

        Args:
            index_name: str

        Returns:
            PineconeVectorStore
        """

        load_dotenv()

        pc_api_key = getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=pc_api_key)

        # Ensure provide vector store name exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name not in existing_indexes:
            raise NameError('provided vector store name does not exist')
        pc_index = pc.Index(index_name)

        # TODO: move model param as class property with small as default
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")   
        vector_store = PineconeVectorStore(index=pc_index, embedding=embeddings)
        
        return vector_store
    
    def retrieve_similar(self, messages:List[dict], k:int= 3) -> List[str]:
        """Wrapper over similarity search

        Args:
            messages: List[dict]
        

        Returns:
            List[str]
        """
        
        load_dotenv()
        latest_message:str = messages[-1]['content']

        try:
            context = self.vector_store.similarity_search(latest_message, k=k)
            # Extract string content from each document
            return ", ".join([document_content.page_content for document_content in context])
        
        except exceptions.PineconeApiException as err_pinecone:
            logging.error(f'Error occurred {err_pinecone}')
            return None
        