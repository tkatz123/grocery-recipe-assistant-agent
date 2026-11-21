# grocery-recipe-assistant-agent

IST 688 final project, Group 3: a LangChain agent that plans grocery trips on a budget (live Kroger prices) and suggests recipes (RAG over TheMealDB).

## Layout

```
agent/       LangChain agent and memory (milestones 5, 7, 8)
tools/       Agent tools: recipe retrieval (retrieve.py) and Kroger prices (krogertool.py)
pipeline/    One-time data build: fetch TheMealDB recipes, load them into Chroma
data/        recipes.json (recipe store) and chroma_db/ (vector index)
.streamlit/  secrets.toml, local only and gitignored
```

## Setup

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (never commit it):

```
[kroger]
client_id = "..."
client_secret = "..."
```

## Running

**Run everything from the repo root as a module** (`python -m folder.file`). Running a file directly by path breaks the imports between folders, and `st.secrets` looks for `.streamlit/` in the folder you run from.

```
python -m pipeline.fetch_recipes        # rebuild data/recipes.json (only if refreshing data)
python -m pipeline.load_chroma          # rebuild data/chroma_db (only if refreshing data)
python -m tools.retrieve --ingredients chicken rice garlic
python -m tools.retrieve --query "creamy pasta" --area Italian
```

`data/chroma_db/` is committed so nobody needs to rebuild it.
