"""Top-level task functions executed inside the job subprocess (see api/jobs.py).

Each heavy pipeline stage is wrapped here so it can be pickled by the ``spawn`` start
method and run in a child process that the parent can terminate on demand. Imports of the
heavy pipeline modules are deferred into the function bodies so importing this module (which
the child does at startup) stays cheap.
"""
from dotenv import load_dotenv

# The child process is spawned fresh, so it must load .env itself for LLM credentials.
load_dotenv()


def ingest_task(table=None, query=None, credentials=None, file_path=None) -> dict:
    from pipeline import ingest
    return ingest.run(table=table, query=query, credentials=credentials, file_path=file_path)


def eda_task(run_id: str) -> dict:
    from pipeline import pre_clean_eda
    return pre_clean_eda.run(run_id)


def clean_task(run_id: str, use_llm: bool = True) -> dict:
    """Decide the cleaning recipe (LLM or rule-based) then execute it on the parquet.
    Returns the merged agent + cleaner result (without run_id/status, which the route adds)."""
    from agents import cleaning_agent
    from pipeline import cleaner
    agent = cleaning_agent.run(run_id, use_llm=use_llm)
    clean = cleaner.run(run_id)
    out = {
        "recipe_source": agent["recipe_source"],
        "recipe_error": agent["recipe_error"],
        "recipe": agent["recipe"],
        "llm_model": agent.get("llm_model"),
        "llm_response": agent.get("llm_response"),
    }
    out.update({k: v for k, v in clean.items() if k != "run_id"})
    return out


def validate_task(run_id: str) -> dict:
    from pipeline import validation_gate
    return validation_gate.run(run_id)
