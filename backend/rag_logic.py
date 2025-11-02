'''Logic Responsible for ingesting PDF into Pinecone VectorStore'''

from langchain_community.document_loaders import PyPDFLoader #Load PDF as Langchain Document
from langchain_text_splitters import RecursiveCharacterTextSplitter #Split Document into Chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Embedding Model
from langchain_pinecone import PineconeVectorStore #VectorStore

#env variable imports
import os
from dotenv import load_dotenv
load_dotenv()

'''Function responsible for ingesting PDF into Pinecone VectorStore'''
def ingestion():
    #Load PDF as Langchain Document Object
    loader = PyPDFLoader("backend/data/GT_book.pdf")
    document = loader.load()

    #Split Document into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200 
    )
    texts = text_splitter.split_documents(document)

    #Create Text Embeddings
    #Initialize Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    #Create Embeddding and Store into Pincone VectorStore
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print('Complete')

ingestion() #Function Call