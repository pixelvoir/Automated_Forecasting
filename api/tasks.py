"""Top-level task functions executed inside the job subprocess (see api/jobs.py).

Each heavy pipeline stage is wrapped here so it can be pickled by the ``spawn`` start
method and run in a child process that the parent can terminate on demand. Imports of the
heavy pipeline modules are deferred into the function bodies so importing this module (which
the child does at startup) stays cheap.

Every task brackets its work with progress.stage_start/stage_finish — the UI's live
progress rail reads ``runs/{id}/progress.json``, and the filesystem is the only channel
that crosses the subprocess boundary (a kill leaves a stranded "running" entry, which
the /progress route reconciles via the job-alive flag).
"""
from pipeline import resource_limits

# Must run before the first numpy/pandas import in this process — the spawn child
# imports this module at startup, so the thread caps land before any BLAS loads.
resource_limits.apply()

from dotenv import load_dotenv

from pipeline import progress

# The child process is spawned fresh, so it must load .env itself for LLM credentials.
# override=True: the child inherits the parent server's env, which holds whatever the
# .env said at SERVER startup — without override, editing .env (e.g. rotating an API
# key) silently keeps the stale key until the server restarts.
load_dotenv(override=True)


def _staged(run_id: str, stage: str, fn, detail: str | None = None):
    """Run fn() bracketed by progress start/finish; failure is recorded then re-raised."""
    progress.stage_start(run_id, stage, detail=detail)
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 — record, then let jobs.py report it
        progress.stage_finish(run_id, stage, status="failed",
                              detail=f"{stage.replace('_', ' ')} failed: {exc}")
        raise
    progress.stage_finish(run_id, stage)
    return result


def ingest_task(table=None, query=None, credentials=None, file_path=None) -> dict:
    from pipeline import ingest
    # No run_id exists until ingest.run returns — bracket the finish only.
    result = ingest.run(table=table, query=query, credentials=credentials, file_path=file_path)
    run_id = result.get("run_id")
    if run_id:
        progress.stage_start(run_id, "ingest", detail="data ingestion")
        progress.stage_finish(run_id, "ingest", detail="data ingested")
    return result


def eda_task(run_id: str) -> dict:
    from pipeline import pre_clean_eda
    return _staged(run_id, "pre_clean_eda", lambda: pre_clean_eda.run(run_id),
                   detail="profiling data (pre-clean EDA)")


def clean_task(run_id: str, use_llm: bool = True, skip_cleaning: bool = False) -> dict:
    """Decide the cleaning recipe (LLM or rule-based; skip_cleaning → pass-through
    essentials-only recipe) then execute it on the parquet. Returns the merged agent +
    cleaner result (without run_id/status, which the route adds)."""
    from agents import cleaning_agent
    from pipeline import cleaner

    def _run():
        agent = cleaning_agent.run(run_id, use_llm=use_llm, skip=skip_cleaning)
        progress.stage_update(run_id, "clean",
                              f"recipe decided ({agent['recipe_source']}) — executing")
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

    return _staged(run_id, "clean", _run, detail="cleaning")


def validate_task(run_id: str) -> dict:
    from pipeline import validation_gate
    return _staged(run_id, "validate", lambda: validation_gate.run(run_id),
                   detail="validation gate")


def forecast_intent_task(run_id: str, use_llm: bool = True) -> dict:
    """Stage 2.5: rule-based intent detection + optional LLM refinement. Suggestion
    quality degrades gracefully — an LLM failure leaves the rule suggestions standing."""
    from pipeline import forecast_intent
    from agents import forecast_intent_agent

    def _run():
        forecast_intent.detect(run_id)
        progress.stage_update(run_id, "intent", "rule suggestions ready — LLM refinement")
        return forecast_intent_agent.refine(run_id, use_llm=use_llm)

    return _staged(run_id, "intent", _run, detail="detecting forecast intent")


def forecast_eda_task(run_id: str) -> dict:
    """Stage 4: full forecasting EDA on the user-confirmed intent."""
    from pipeline import forecasting_eda
    return _staged(run_id, "forecast_eda", lambda: forecasting_eda.run(run_id),
                   detail="forecast EDA")


def model_select_task(run_id: str, use_llm: bool = True) -> dict:
    """Stage 5: the rule engine always decides first; the LLM (when enabled) may
    override it. An LLM failure leaves the rule-engine pick standing."""
    from agents import model_selector_agent
    return _staged(run_id, "model_select",
                   lambda: model_selector_agent.run(run_id, use_llm=use_llm),
                   detail="model selection")


def train_task(run_id: str, models: list, confirm_heavy: bool = False,
               profile: str | None = None) -> dict:
    """Stages 6+7: build the EDA-seeded features (materialize the ML matrix only when an
    ML model was selected), then train + backtest the chosen models. No LLM, no DB.
    The 'heavy' cost gate is enforced in the route before this launches, and again here
    (defense in depth). `profile` optionally overrides training.accuracy_profile."""
    from pipeline import feature_builder, trainer

    def _run():
        build_ml = any(m in trainer.registry.ML_IDS for m in (models or []))
        progress.stage_update(run_id, "train", "building features (Stage 6)")
        fr = feature_builder.run(run_id, build_ml_matrix=build_ml)
        result = trainer.run(run_id, models=models, confirm_heavy=confirm_heavy,
                             profile=profile)
        result["feature_report"] = fr.get("report")
        return result

    return _staged(run_id, "train", _run, detail="training")
