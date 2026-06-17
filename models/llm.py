import langchain
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_hyperbrowser import HyperbrowserLoader
# from langchain.chains import RetrievalQAWithSourcesChain as RetrievalQA
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import UnstructuredURLLoader

load_dotenv()

def process_urls(urls):
    """This function scrapes data from a url and stores it in a vector DB
       :param urls: input urls
       :return: 
    """
    
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

RecursiveCharacterTextSolitter(
    seperators=["\n\n", "\n", ".", " "]
    chunk_size=CHUNK_SIZE
)

if __name__ == "__main__":
    urls = []
    
    process_urls(urls)


# api_key = os.getenv("GROQ_API_KEY")

# print(api_key)
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0, 
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=api_key
    # other params...
)

prompt = llm.invoke("What year did all countries of Africa gain independence")
print(prompt)


loader = HyperbrowserLoader(
    urls="https://jumbo.nl",
    api_key=api_key,
)

docs = loader.load()
docs[0]
