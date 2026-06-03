.PHONY: help up down build logs health config shell data ingest index ask baseline lint fmt

# Run a CLI command inside the app container.
DC := docker compose
EXEC := $(DC) exec app anchored

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Build + start Elasticsearch and the app
	$(DC) up --build -d
	@echo "Waiting for the stack to be healthy..."
	@$(DC) ps

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

baseline: ## Run the retrieval eval + write BASELINE.md (#5)
	$(EXEC) baseline

lint: ## Lint with ruff
	$(DC) exec app ruff check anchored

fmt: ## Format with ruff
	$(DC) exec app ruff format anchored
