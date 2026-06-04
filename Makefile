.PHONY: help up down build logs health config shell data ingest index ask baseline lint fmt

# Run a CLI command inside the app container.
DC := docker compose
EXEC := $(DC) exec app anchored

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Build + start Elasticsearch and the app (UI at http://localhost:8000)
	$(DC) up --build -d
	@echo "Waiting for the stack to be healthy..."
	@$(DC) ps
	@echo "\nDemo UI → http://localhost:8000"

ui: ## Open the demo UI in a browser
	@open http://localhost:8000 || xdg-open http://localhost:8000 || echo "Open http://localhost:8000"

down: ## Stop and remove containers
	$(DC) down

build: ## Build the app image
	$(DC) build

logs: ## Tail logs
	$(DC) logs -f

health: ## Check Elasticsearch connectivity from the app
	$(EXEC) health

config: ## Print resolved configuration
	$(EXEC) config

shell: ## Open a shell in the app container
	$(DC) exec app bash

data: ## Download + verify the CUAD corpus (#2)
	$(EXEC) data

ingest: ## Process raw contracts -> chunks.jsonl (#3)
	$(EXEC) ingest

index: ## Embed chunks + build the Elasticsearch index (#3)
	$(EXEC) index

ask: ## Ask a question, e.g. make ask Q="termination clause?" (#3)
	$(EXEC) ask "$(Q)"

trace: ## Inspect retrieval traces (summary + recent events)
	$(EXEC) trace

notebook: ## Rebuild + execute the exploration notebook (charts baked in)
	$(DC) exec app python notebooks/build_explore.py
	$(DC) exec app jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=300 notebooks/01_explore.ipynb

baseline: ## Run the retrieval eval + write BASELINE.md (#5)
	$(EXEC) baseline

lint: ## Lint with ruff
	$(DC) exec app ruff check anchored

fmt: ## Format with ruff
	$(DC) exec app ruff format anchored
