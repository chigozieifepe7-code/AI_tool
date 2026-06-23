import langchain
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# from langchain.chains import RetrievalQAWithSourcesChain as RetrievalQA
# from langchain_hyperbrowser import HyperbrowserLoader

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "research"

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(
            model="qwen/qwen3-32b", # (This Model is being decommisioned from July 17. Change Model)
            temperature=0, 
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
            api_key=api_key
            # other params...
        )
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name = "EMBEDDING_MODEL",
            model_kwargs = {"trust_remote_code" : True}
        )
        vector_store = Chroma(
            collection_name = COLLECTION_NAME,
            persist_directory = str(VECTORSTORE_DIR),
            embedding_function = ef
        )

def process_urls(urls):
    """This function scrapes data from a url and stores it in a vector DB
       :param urls: input urls
       :return: 
    """
    print("Initialize components")
    # This checks if the vecto db has been initialized previously 
    initialize_components()
    # This resets the Db and aggregates the new urls together with the previous ones
    vector_store.reset_collection()
        
    print("Load data")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    print("Split text")
    text_splitter = RecursiveCharacterTextSplitter(
            seperators=["\n\n", "\n", ".", " "],
            chunk_size=CHUNK_SIZE
        )
    # WE CALL THE CHUNKS DOCS HERE
    docs = text_splitter.split_documents(data)

    print("Add docs to vector db")
    # Get unique ids for each document chunk
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    


if __name__ == "__main__":
    urls = [
        
        
    ]
    
    process_urls(urls)
    results = vector_store.similarity_search(
        "30 year mortgage rate",
        k=2
    )
    print(results)


# api_key = os.getenv("GROQ_API_KEY")

# print(api_key)







    
    
    
    
    
-----------------------------------------------------------------------------------------
    
#     prompt = llm.invoke("What year did all countries of Africa gain independence")
#     print(prompt)


# loader = HyperbrowserLoader(
#     urls="https://jumbo.nl",
#     api_key=api_key,
# )

# docs = loader.load()
# docs[0]

# from firecrawl import Firecrawl

# firecrawl = Firecrawl(
  # No API key needed to get started — add one for higher rate limits:
  # api_key="fc-YOUR-API-KEY",
# )

# Scrape a website:
# doc = firecrawl.scrape("https://firecrawl.dev", formats=["markdown", "html"])
# print(doc)