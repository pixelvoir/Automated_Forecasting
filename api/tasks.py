"""Top-level task functions executed inside the job subprocess (see api/jobs.py).

Each heavy pipeline stage is wrapped here so it can be pickled by the ``spawn`` start
method and run in a child process that the parent can terminate on demand. Imports of the
heavy pipeline modules are deferred into the function bodies so importing this module (which
the child does at startup) stays cheap.
"""
from dotenv import load_dotenv

# The child process is spawned fresh, so it must load .env itself for LLM credentials.
# override=True: the child inherits the parent server's env, which holds whatever the
# .env said at SERVER startup — without override, editing .env (e.g. rotating an API
# key) silently keeps the stale key until the server restarts.
load_dotenv(override=True)


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


def forecast_intent_task(run_id: str, use_llm: bool = True) -> dict:
    """Stage 2.5: rule-based intent detection + optional LLM refinement. Suggestion
    quality degrades gracefully — an LLM failure leaves the rule suggestions standing."""
    from pipeline import forecast_intent
    from agents import forecast_intent_agent
    forecast_intent.detect(run_id)
    return forecast_intent_agent.refine(run_id, use_llm=use_llm)


def forecast_eda_task(run_id: str) -> dict:
    """Stage 4: full forecasting EDA on the user-confirmed intent."""
    from pipeline import forecasting_eda
    return forecasting_eda.run(run_id)


def model_select_task(run_id: str, use_llm: bool = True) -> dict:
    """Stage 5: the rule engine always decides first; the LLM (when enabled) may
    override it. An LLM failure leaves the rule-engine pick standing."""
    from agents import model_selector_agent
    return model_selector_agent.run(run_id, use_llm=use_llm)


def train_task(run_id: str, models: list, confirm_heavy: bool = False) -> dict:
    """Stages 6+7: build the EDA-seeded features (materialize the ML matrix only when an
    ML model was selected), then train + backtest the chosen models. No LLM, no DB.
    The 'heavy' cost gate is enforced in the route before this launches, and again here
    (defense in depth)."""
    from pipeline import feature_builder, trainer
    build_ml = any(m in trainer.registry.ML_IDS for m in (models or []))
    fr = feature_builder.run(run_id, build_ml_matrix=build_ml)
    result = trainer.run(run_id, models=models, confirm_heavy=confirm_heavy)
    result["feature_report"] = fr.get("report")
    return result
