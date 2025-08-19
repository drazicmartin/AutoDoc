import logging
import os

import chromadb
from tqdm import tqdm
import typer
import yaml
from firecrawl import FirecrawlApp, ScrapeOptions
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from utils.ollama import OllamaEmbedding
from utils.webreader import FireCrawlWebReader

# Set up logger
logger = logging.getLogger(__name__)

def setup_logger(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    
    # Configure your logger only
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent messages from propagating to root logger

    # Suppress other libraries' logging below WARNING
    logging.getLogger().setLevel(logging.WARNING)
    for lib in ['chromadb', 'firecrawl', 'llama_index', 'some_other_lib']:
        logging.getLogger(lib).setLevel(logging.WARNING)

def is_running_in_docker():
    return os.path.exists("/.dockerenv")

FIRECRAWL_URL = "http://localhost:3002" if not is_running_in_docker() else "http://firecrawl-api:3002"
OLLAMA_URL = "http://localhost:11434" if not is_running_in_docker() else "http://ollama:11434"
CHROMA_URL = "localhost" if not is_running_in_docker() else "chroma"
CHROMA_PORT = 8000

ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_URL,
    ollama_additional_kwargs={"mirostat": 0},
)

chroma_client = chromadb.HttpClient(host=CHROMA_URL, port=CHROMA_PORT)

def main(
    config_path: str = "docs.yml",
    ow: bool = False,  # overwrite existing collections
    limit: int = 1000,
    debug: bool = False,  # Add debug parameter
    mode: str = "async_crawl",
):
    setup_logger(debug)
    logger.debug("Starting documentation processing")

    # open docs
    with open(config_path, "r") as f:
        docs = yaml.safe_load(f)
    logger.debug("Configuration file loaded successfully")

    # scrape docs
    for lib_name, lib in docs.items():
        logger.debug(f"Processing library: {lib_name}")
        for version in lib["versions"]:
            url = lib["url"].format(version=version)
            logger.info(f"Scraping {lib_name} version {version} from {url}")

            collection_name = f"{lib_name}_{version}"
            all_collections: list = chroma_client.list_collections()

            if collection_name in all_collections and ow:
                chroma_client.delete_collection(collection_name)
                all_collections.remove(collection_name)
                logger.debug(f"Deleted collection {collection_name}")
                continue

            if collection_name in all_collections:
                logger.info(f"Collection {collection_name} already exists, skipping...")
                continue
            
            chroma_collection = chroma_client.get_or_create_collection(collection_name, metadata={"lib_name": lib_name, "version": version})
            logger.debug(f"Created/Retrieved collection: {collection_name}")

            firecrawl_reader = FireCrawlWebReader(
                api_url=FIRECRAWL_URL,
                api_key="",
                mode=mode,
                params=dict(
                    scrape_options=ScrapeOptions(
                        formats=['markdown'],
                    ),
                    include_paths=[f".*/{version}/.*"],
                    allow_backward_links=True,
                    allow_external_links=False,
                    limit=limit,
                )
            )
            logger.debug("FireCrawl reader initialized")

            lib_documents = firecrawl_reader.load_data(url=url)
            logger.debug(f"Loaded {len(lib_documents)} documents")

            lib_documents = [doc for doc in lib_documents if doc.text.strip() != '']
            logger.debug(f"Filtered documents: {len(lib_documents)}")

            for doc in lib_documents:
                doc.metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool, type(None)))}
            logger.debug("Document metadata filtered")

            parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=100)
            nodes = parser.get_nodes_from_documents(lib_documents)
            logger.debug(f"Documents chunked into {len(nodes)} nodes")

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            logger.debug("Vector store initialized")

            batch_size = 512
            for i in tqdm(range(0, len(nodes), batch_size)):
                embeddings = ollama_embedding.get_text_embedding_batch([node.text for node in nodes[i:i + batch_size]])
                for node, embedding in zip(nodes[i:i + batch_size], embeddings):
                    node.embedding = embedding
                vector_store.add(nodes[i:i + batch_size])
            logger.debug("Node embeddings generated")

            logger.debug(f"{len(nodes)} Nodes added to vector store")

            logger.info(f"Completed processing {lib_name} version {version}")
    
    logger.info("All processing completed successfully!")

if __name__ == "__main__":
    typer.run(main)