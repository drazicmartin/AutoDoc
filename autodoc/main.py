import chromadb
import yaml
from firecrawl import FirecrawlApp, ScrapeOptions
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from utils.ollama import OllamaEmbedding
from utils.webreader import FireCrawlWebReader

app = FirecrawlApp(
    api_key="",
    api_url="http://localhost:3002"
)

ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

chroma_client = chromadb.EphemeralClient()

if __name__ == "__main__":
    # open docs
    with open("docs.yml", "r") as f:
        docs = yaml.safe_load(f)

    # scrape docs
    for lib_name, lib in docs.items():
        for version in lib["versions"]:
            url = lib["url"].format(version=version)
            print(f"Scraping {lib_name} version {version} from {url}")

            collection_name = f"{lib_name}_{version}"
            chroma_collection = chroma_client.create_collection(collection_name, metadata={"lib_name": lib_name, "version": version})

            firecrawl_reader = FireCrawlWebReader(
                api_url="http://localhost:3002/",
                api_key="",
                mode="crawl",  # Choose between "crawl" and "scrape" for single page scraping
                params=dict(
                    scrape_options=ScrapeOptions(formats=['markdown']),
                    include_paths=[url, f".*/{version}/.*"],
                    allow_backward_links=True,
                    allow_external_links=False,
                    limit=50
                )
            )

            lib_documents = firecrawl_reader.load_data(url=url)

            for doc in lib_documents:
                # filter out all metadata that are not str, int, float, bool, or None
                doc.metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool, type(None)))}

            # set up ChromaVectorStore and load in data
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                lib_documents, storage_context=storage_context, embed_model=ollama_embedding
            )

            # query_engine = index.as_query_engine()
            # response = query_engine.query("What did the author do growing up?")

            retriever_engine = index.as_retriever()
            response = retriever_engine.retrieve("torch")
            for result in response[:3]:
                print(result)
                print(result.score)

            breakpoint()

            # Load documents from a single page URL
            # documents = firecrawl_reader.load_data(url)

            # index = SummaryIndex.from_documents(documents)

            # Set Logging to DEBUG for more detailed outputs

            breakpoint()
            