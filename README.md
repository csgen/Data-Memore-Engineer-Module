# Scraper, Preprocessing & Memory Modules

Data ingestion and memory layer for the Agentic AI News Fact-Checking system.

## Architecture

```
News Sources --> Scraper Agent --> Preprocessing Agent --> Memory Agent --> Downstream Agents
                (Tavily, RSS,    (Claim isolation,       (ChromaDB +      (Fact-Check,
                 Telegram)           Entity extraction,      Neo4j)           Entity Tracker,
                                   VLM captioning)                          Prediction)
```

### Modules

- **Scraper Agent** — fetches news from Tavily, RSS feeds (BBC, Reuters, AP), and Telegram. Deduplicates via SHA-256 content hashing.
- **Preprocessing Agent** — extracts falsifiable claims (LLM), entities (spaCy + LLM), and image captions (GPT-4o vision). Outputs structured JSON.
- **Memory Agent** — dual-database wrapper (ChromaDB for semantic search, Neo4j for knowledge graph). Single facade API for all agents.

### Databases/Storage (Cloud-Hosted)

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| Neo4j Aura | Knowledge Graph (entities, claims, relationships) | 200k nodes / 400k relationships |
| ChromaDB Cloud | Vector DB (semantic search on claims, articles, captions) | Sufficient for project scale |
| Cloudflare R2 | Object storage for Telegram images | 10 GB storage / 10M reads per month |

### Why Telegram + Object Storage

The original plan used Reddit as the social media source, but Reddit's API access approval process was not feasible within our project timeline. We switched to Telegram, which provides a similar role: unverified breaking news, rumors, and tip-offs from public channels -- the kind of claims our fact-checking pipeline is designed to verify.

Unlike news articles (Tavily/RSS) where images are already hosted at public URLs, Telegram images are media attachments that need to be downloaded. We upload them to Cloudflare R2 to get a public URL, so the rest of the pipeline (VLM captioning, frontend display) works unchanged.

## Setup

### 1. Prerequisites

- Docker Desktop installed and running
- Neo4j Aura Free account ([aura.neo4j.io](https://aura.neo4j.io))
- ChromaDB Cloud account ([trychroma.com](https://trychroma.com))
- OpenAI API key
- Tavily API key (optional, 1000 free credits/month at [tavily.com](https://tavily.com))
- Telegram API credentials (optional, get at [my.telegram.org](https://my.telegram.org))
- Cloudflare R2 account (optional, free 10GB for image storage)

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Build and Run

```bash
# Build the Docker image and start the container
docker-compose build && docker-compose up -d

# Initialize Neo4j schema (constraints + indexes)
docker-compose exec app python scripts/init_neo4j.py

# Seed known news sources into Neo4j
docker-compose exec app python scripts/seed_sources.py

# Run the full pipeline (scrape -> preprocess -> store)
docker-compose exec app python -m src.pipeline
```

### 4. Run Tests

```bash
docker-compose exec app pytest tests/ -v
```

## Teammate Integration

Teammates import `MemoryAgent` to read/write data:

```python
from src.memory.agent import MemoryAgent
from src.models.verdict import Verdict
from src.config import settings

memory = MemoryAgent(settings)

# Query similar claims
results = memory.search_similar_claims("Tesla recalled 500k vehicles", top_k=5)

# Write a verdict
verdict = Verdict(verdict_id="vrd_...", claim_id="clm_...", ...)
memory.add_verdict(verdict)

# Get entity credibility context
context = memory.get_entity_context(claim_id="clm_...")

# Find trending entities
trending = memory.get_trending_entities(since=one_week_ago, limit=10)
```

## Project Structure

```
src/
  config.py              # Environment config (pydantic-settings)
  id_utils.py            # UUID generation with type prefixes
  pipeline.py            # End-to-end orchestration
  models/                # Pydantic data models (shared contract)
  memory/                # Memory Agent (ChromaDB + Neo4j facade)
  scraper/               # Scraper Agent (Tavily, RSS, Telegram)
  preprocessing/         # Preprocessing Agent (claims, entities, captions)
scripts/
  init_neo4j.py          # Create DB constraints + indexes
  seed_sources.py        # Seed SOURCE nodes
data/
  sources.json           # Known news sources with credibility scores
tests/                   # Unit and integration tests
```
