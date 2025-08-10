# AutoDoc 

> A Very simple offline documentation context provider for LLM via MCP

> How I hope it will works : Use web crawlers to extract accurate documentation, embed each chunks with ollama embedding models, store in vector database, then an MCP will retrieve most relevant inforamtion based on natural language prompt provided by LLM.

## Stacks
- llama-index
- chroma
- firecrawl
- ollama

## Run Dev

```bash
cp docker/.env.exemple docker/.env
docker compose -f docker/compose-fire.yml

pip install -r requirements.txt
python autodoc/main.py
```