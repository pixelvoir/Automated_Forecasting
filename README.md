# Automated Forecasting Agent

Infrastructure scaffold for an automated forecasting pipeline.

## Folder Structure

```text
.
├── agents/
├── api/
├── config/
├── data/
├── frontend/
├── models_lib/
├── pipeline/
├── runs/
├── tests/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

## Prerequisites
- Python 3.11+

## Quickstart

This is the infrastructure scaffold step; pipeline stages are implemented incrementally in later steps. Async job handling and external artifact storage were deliberately left out until there is an actual need for them.

**One-time setup** (creates `.venv` and installs dependencies):
```bat
setup_venv.bat
```

**Start the dev server** (auto-reloads on file changes):
```bat
run_dev.bat
```

**Health check:**
```
curl http://localhost:8000/health
```

**Start the Dash UI** (in a second terminal, after the API is running):
```bat
run_frontend.bat
```
Then open `http://localhost:8050` in your browser.

> **Note on `prophet`:** It requires C++ build tools (`pystan` backend) and is commented out of `requirements.txt`. See the comment there for install options. All other dependencies install cleanly via pip.
