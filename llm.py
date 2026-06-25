import langchain
import os
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain


load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "research"

llm = None
vector_store = None
api_key = os.getenv("GROQ_API_KEY")


def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(
            # (This Model is being decommisioned from July 17. Change Model)
            model="qwen/qwen3-32b",
            temperature=0,
            max_retries=2,
            api_key=api_key
        )
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=ef
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
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    # WE CALL THE CHUNKS DOCS HERE
    docs = text_splitter.split_documents(data)

    print("Add docs to vector db")
    # Get unique ids for each document chunk
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector DB is not initialized")
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result["answer"], sources


if __name__ == "__main__":
    urls = [
        "https://www.tue.nl/en/working-at-tue/vacancy-overview/engineering-doctorate-engd-programs-in-data-software-systems-design",
        "https://www.cnbc.com/2026/06/04/higher-mortgage-rates-push-application-denial-rates-up.html",
        "https://www.cnbc.com/2026/06/23/tech-stocks-sell-off-mag7-samsung-sk-hynix.html"
    ]

    process_urls(urls)
    answer, sources = generate_answer(
        "What is the need for this Job vacancy? What are the skills needed to complete the tasks in this vacancy? What skills will this program give me?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")


# api_key = os.getenv("GROQ_API_KEY")

# print(api_key)


# -----------------------------------------------------------------------------------------

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
