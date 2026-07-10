"""Shared tiny UI primitives for the tab views (pure rendering, no callbacks).

Views import from here; callbacks.py doesn't need to — no circularity.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


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
                                      style={"fontSize": "0.72rem", "color": "#475569"}))
    return html.Details([
        html.Summary(summary_kids,
                     style={"cursor": "pointer", "color": "#94a3b8", "fontSize": "0.82rem",
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
                   style={"color": "#64748b", "fontSize": "0.7rem"}),
        dcc.DatePickerSingle(
            id=picker_id, date=None, min_date_allowed=day,
            placeholder="pick end date", display_format="YYYY-MM-DD",
            className="horizon-datepicker"),
        html.Small(id=note_id, className="d-block",
                   style={"color": "#64748b", "fontSize": "0.7rem"}),
    ]
