# AutoDoc 

> A simple offline documentation context provider for LLMs via MCP, designed as an alternative to context7

AutoDoc extracts and processes documentation to enhance LLM interactions through vector-based retrieval.

## How It Works
1. Web crawlers extract documentation
2. Content is embedded using Ollama models
3. Embeddings are stored in a vector database
4. MCP retrieves relevant information based on natural language prompts

## Tech Stack
- llama-index: For document processing
- chroma: Vector database
- firecrawl: Web crawling
- ollama: Embedding models

## Quick Start

### Windows
```powershell
# Setup environment
copy docker\.env.example docker\.env

# Start services
$env:COMPOSE_FILE="docker\compose-fire.yml;docker\compose-chroma.yml"
# or with fire simple
$env:COMPOSE_FILE="docker/compose-fire-simple.yml;docker/compose-chroma.yml"
docker compose up -d

# Install and run
python -m pip install -r requirements.txt
python autodoc\main.py
```

### Linux
```bash
# Setup environment
cp docker/.env.example docker/.env

# Start services
COMPOSE_FILE="docker/compose-fire.yml:docker/compose-chroma.yml"
# or with fire simple
COMPOSE_FILE="docker/compose-fire-simple.yml:docker/compose-chroma.yml"
docker compose up -d

# Install and run
pip install -r requirements.txt
python autodoc/main.py
```