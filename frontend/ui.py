"""Shared tiny UI primitives for the tab views (pure rendering, no callbacks).

Views and callbacks.py import from here — this module imports no view, no circularity.
Also home to the pipeline progress renderers (stage rail, log panel, stage cards) used
by the 3-section layout.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def fmt_duration(seconds) -> str:
    s = float(seconds or 0)
    if s < 60:
        return f"{max(int(round(s)), 1)}s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# ── Pipeline stage definitions (order = the rail's render order) ──────────────
# Keys match pipeline/progress.py STAGES exactly.

STAGE_DEFS = [
    ("ingest",        "Ingest",        "bi-cloud-download"),
    ("pre_clean_eda", "Profile",       "bi-search"),
    ("intent",        "Intent",        "bi-crosshair"),
    ("clean",         "Clean",         "bi-scissors"),
    ("validate",      "Validate",      "bi-shield-check"),
    ("forecast_eda",  "Forecast EDA",  "bi-graph-up"),
    ("model_select",  "Model Select",  "bi-cpu"),
    ("train",         "Train",         "bi-lightning-fill"),
    ("report",        "Report",        "bi-file-earmark-text"),
]

_STATUS_ICON = {
    "pending":   ("bi-circle", "stage-ico--pending"),
    "queued":    ("bi-three-dots", "stage-ico--pending"),
    "running":   ("bi-arrow-repeat spin", "stage-ico--running"),
    "done":      ("bi-check-circle-fill", "stage-ico--done"),
    "failed":    ("bi-x-circle-fill", "stage-ico--failed"),
    "cancelled": ("bi-slash-circle", "stage-ico--cancelled"),
}

_BADGE_COLOR = {"pending": "secondary", "queued": "secondary", "running": "info",
                "done": "success", "failed": "danger", "cancelled": "warning"}


def stage_badge(status: str | None, seconds=None):
    """The small status chip in a stage card's summary row."""
    status = status or "pending"
    text = status if seconds is None or status != "done" else f"done · {fmt_duration(seconds)}"
    return dbc.Badge(text, color=_BADGE_COLOR.get(status, "secondary"),
                     className="ms-2 stage-badge", pill=True)


def _rail_row(key, label, icon, entry, sub_rows=None):
    status = (entry or {}).get("status") or "pending"
    ico, ico_cls = _STATUS_ICON.get(status, _STATUS_ICON["pending"])
    right = ""
    if status == "done" and (entry or {}).get("seconds") is not None:
        right = fmt_duration(entry["seconds"])
    elif status == "running":
        eta = (entry or {}).get("eta_seconds")
        right = f"~{fmt_duration(eta)}" if eta else "running…"
    detail = (entry or {}).get("detail") if status == "running" else None
    kids = [html.Div([
        html.I(className=f"bi {ico} stage-ico {ico_cls}"),
        html.Span(label, className="stage-label"),
        html.Span(right, className="stage-eta"),
    ], className=f"stage-row stage-row--{status}")]
    if detail:
        kids.append(html.Div(detail, className="stage-detail"))
    if sub_rows:
        kids.extend(sub_rows)
    return html.Div(kids)


def _model_sub_rows(models: list) -> list:
    rows = []
    for m in models or []:
        status = m.get("status", "pending")
        ico, ico_cls = _STATUS_ICON.get(status, _STATUS_ICON["pending"])
        bits = []
        if m.get("mase") is not None:
            bits.append(f"MASE {m['mase']:.3f}")
        if status == "done" and m.get("seconds") is not None:
            bits.append(fmt_duration(m["seconds"]))
        elif status == "running" and m.get("eta_seconds"):
            bits.append(f"~{fmt_duration(m['eta_seconds'])}")
        rows.append(html.Div([
            html.I(className=f"bi {ico} stage-ico {ico_cls}"),
            html.Span(m.get("id", "?"), className="stage-label"),
            html.Span(" · ".join(bits), className="stage-eta"),
        ], className=f"stage-row stage-row--sub stage-row--{status}"))
    return rows


def stage_rail(statuses: dict) -> html.Div:
    """The vertical pipeline progress rail. `statuses` = merged per-stage dicts
    (progress.json entries, store-derived fallbacks and queued markers)."""
    rows = []
    for key, label, icon in STAGE_DEFS:
        entry = statuses.get(key) or {}
        subs = _model_sub_rows(entry.get("models")) if key == "train" else None
        rows.append(_rail_row(key, label, icon, entry, subs))
    return html.Div(rows, className="stage-rail")


def log_panel(events: list) -> html.Div:
    """Scrolling pipeline log — newest last (auto-scrolled to the bottom via CSS
    column-reverse trick is unreliable; we simply render newest-last and cap height)."""
    if not events:
        lines = [html.Div("No pipeline activity yet.", className="log-line log-line--empty")]
    else:
        lines = [
            html.Div([
                html.Span(str(e.get("t", ""))[11:19], className="log-t"),
                html.Span(e.get("stage", ""), className="log-stage"),
                html.Span(e.get("msg", "")),
            ], className="log-line")
            for e in events
        ]
    return html.Div(lines, className="pipeline-log")


def stage_card(key: str, title: str, icon: str, body, open=False):
    """A Pipeline-section expander wrapping one stage's full results body. The chrome is
    static (defined once in layout.py); only the badge span (`stage-badge-{key}`) and the
    body div inside are updated by callbacks."""
    return html.Details([
        html.Summary([
            _cicon(icon, fontSize="0.85rem"),
            html.Span(title, className="stage-card-title"),
            html.Span(id=f"stage-badge-{key}", className="ms-2"),
        ], className="stage-card-summary"),
        html.Div(body, className="stage-card-body"),
    ], className="stage-card mb-3", open=open)


def collapse_section(title, children, icon=None, open=False, summary_extra=None):
    """Consistent ``html.Details`` expander for verbose stat tables/sections.

    html.Details over dbc.Accordion deliberately: it is the pattern the codebase already
    uses (raw-JSON payloads, rule traces), it's stateless, and it needs no callback.
    """
    summary_kids = []
    if icon:
        summary_kids.append(_cicon(icon, fontSize="0.8rem"))
    summary_kids.append(html.Span(title))
    if summary_extra:
        summary_kids.append(html.Span(f"  {summary_extra}",
                                      style={"fontSize": "0.72rem", "color": "var(--ink-ghost)"}))
    return html.Details([
        html.Summary(summary_kids,
                     style={"cursor": "pointer", "color": "var(--ink-muted)", "fontSize": "0.82rem",
                            "fontWeight": "600", "userSelect": "none"}),
        html.Div(children, style={"marginTop": "10px"}),
    ], className="mb-3", open=open)


def horizon_datepicker(data_end, picker_id, note_id):
    """The optional "…or forecast until" date picker rendered under a horizon input.

    Returns a list of components (spread into the column). Empty when the last-actual
    timestamp isn't known (old runs) — the numeric horizon input alone then behaves
    exactly as before. The picker starts empty (date=None): a sync callback converts a
    picked date into the canonical integer horizon; pre-filling would fight the
    mount-fire guard in that callback.
    """
    if not data_end:
        return []
    day = str(data_end)[:10]
    return [
        html.Small("…or forecast until", className="d-block mt-2",
                   style={"color": "var(--ink-faint)", "fontSize": "0.7rem"}),
        dcc.DatePickerSingle(
            id=picker_id, date=None, min_date_allowed=day,
            placeholder="pick end date", display_format="YYYY-MM-DD",
            className="horizon-datepicker"),
        html.Small(id=note_id, className="d-block",
                   style={"color": "var(--ink-faint)", "fontSize": "0.7rem"}),
    ]
