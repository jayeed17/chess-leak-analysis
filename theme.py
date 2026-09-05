"""Theme + components for app.py — Wall Street Journal direction.

    from theme import (inject_theme, page_header, kpi_row, split_rate,
                       SEVERITY, BAR_BASE, board_svg, board_strip,
                       severity_bar, trend_chart)

Palette is newsprint: warm paper, black ink, hairline rules. Colour is reserved
for severity and never used decoratively.

The old board-green was actively misleading — in chess interfaces green means
"good move", so colouring blunder bars green read as praise for the worst moves
in the dataset. Severity now runs ochre -> orange -> red, so worse is visibly
hotter, and the single non-severity accent (slate blue) marks things that went
right.
"""
import chess
import chess.svg

# newsprint
PAPER = "#F7F4ED"
PANEL = "#FFFFFF"
RULE = "#D6D0C4"
HAIR = "#B8B1A3"
INK = "#12100D"
BODY = "#33302A"
MUTED = "#6E685C"

# severity ramp — hotter is worse. Never use decoratively.
SEVERITY = {
    "blunder": "#9B1B21",
    "mistake": "#C4551F",
    "inaccuracy": "#D69A2D",
    "good": "#2E5C7A",
    "neutral": "#6E685C",
}

# neutral bar/line color for charts that aren't themselves a severity ranking --
# a ranking highlights its one worst bar in SEVERITY["blunder"] instead (see
# severity_bar()); a plain series (the trend line) just stays this color.
BAR_BASE = "#5B7C99"

_CSS = f"""
<style>
  html, body, [class*="css"], .stApp, [data-testid="stSidebar"],
  [data-testid="stMarkdownContainer"], .stDataFrame, .stSelectbox,
  .stMultiSelect, button, input, label, .stRadio {{
    font-family: "Times New Roman", Times, serif !important;
  }}
  .stApp {{ background: {PAPER}; color: {BODY}; }}
  [data-testid="stSidebar"] {{ background: {PAPER}; border-right: 1px solid {RULE}; }}

  .masthead {{ text-align: center; margin: 0.4rem 0 0.2rem; }}
  .masthead h1 {{
    font-size: 3.4rem; font-weight: 700; color: {INK};
    letter-spacing: -0.02em; line-height: 1.02; margin: 0;
  }}
  .masthead .byline {{ margin-top: 0.5rem; font-size: 1.02rem; color: {BODY}; }}
  .masthead .byline a {{ color: {SEVERITY['good']}; text-decoration: none;
                         border-bottom: 1px solid {RULE}; }}
  .masthead .rules {{
    border-top: 2px solid {INK}; border-bottom: 1px solid {HAIR};
    height: 4px; margin: 0.9rem 0 0.3rem;
  }}
  .dateline {{
    text-align: center; font-size: 0.9rem; color: {MUTED}; margin-bottom: 1.4rem;
  }}

  h2 {{ font-size: 1.75rem; color: {INK}; font-weight: 700; margin-top: 1.8rem; }}
  h3 {{ font-size: 1.25rem; color: {INK}; font-weight: 700; }}
  [data-testid="stMarkdownContainer"] p {{
    line-height: 1.7; max-width: 72ch; color: {BODY}; font-size: 1.05rem;
  }}
  [data-testid="stCaptionContainer"] p {{
    color: {MUTED}; font-size: 0.93rem; line-height: 1.6; max-width: 72ch;
  }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 2rem; border-bottom: 1px solid {HAIR}; }}
  .stTabs [data-baseweb="tab"] {{ font-size: 1.1rem; color: {MUTED}; padding: 0.4rem 0; }}
  .stTabs [aria-selected="true"] {{ color: {INK}; border-bottom: 3px solid {INK}; }}

  .kpi-row {{
    display: flex; gap: 0; flex-wrap: wrap;
    border-top: 1px solid {HAIR}; border-bottom: 1px solid {HAIR};
    margin: 0.6rem 0 1.8rem; background: {PANEL};
  }}
  .kpi {{
    flex: 1 1 200px; padding: 1.05rem 1.25rem 1.15rem;
    border-right: 1px solid {RULE}; border-top: 3px solid var(--accent);
  }}
  .kpi:last-child {{ border-right: none; }}
  .kpi .label {{ font-size: 0.98rem; color: {MUTED}; margin-bottom: 0.3rem; }}
  .kpi .value {{
    font-size: 3.1rem; font-weight: 700; line-height: 0.98;
    color: {INK}; font-variant-numeric: tabular-nums;
  }}
  .kpi .sub {{
    font-size: 0.86rem; color: {MUTED}; margin-top: 0.4rem;
    font-variant-numeric: tabular-nums;
  }}
  .kpi.thin .sub::after {{ content: " · thin sample"; color: {SEVERITY['blunder']}; }}

  .boards {{ display: flex; gap: 1.1rem; flex-wrap: wrap; margin: 0.8rem 0 1.4rem; }}
  .board-fig {{ flex: 0 1 250px; }}
  .board-fig svg {{ width: 100%; height: auto; border: 1px solid {RULE}; }}
  .board-fig .cap {{
    font-size: 0.88rem; color: {BODY}; line-height: 1.45;
    margin-top: 0.45rem; border-top: 1px solid {RULE}; padding-top: 0.35rem;
  }}
  .board-fig .cap b {{ color: {INK}; }}

  .stDataFrame {{ border: 1px solid {RULE}; }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
"""


def inject_theme(st):
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(st, name, handle, linkedin_url, dateline):
    st.markdown(
        f'<div class="masthead"><h1>Chess Leak Analysis</h1>'
        f'<div class="byline">{name} &middot; {handle} &middot; '
        f'<a href="{linkedin_url}" target="_blank">LinkedIn</a></div>'
        f'<div class="rules"></div></div>'
        f'<div class="dateline">{dateline}</div>',
        unsafe_allow_html=True)


def kpi_row(st, items):
    """items: label, value, sub (optional), severity key, thin (optional bool)."""
    cards = []
    for it in items:
        accent = SEVERITY.get(it.get("severity", "neutral"), SEVERITY["neutral"])
        cls = "kpi thin" if it.get("thin") else "kpi"
        sub = f'<div class="sub">{it["sub"]}</div>' if it.get("sub") else ""
        cards.append(f'<div class="{cls}" style="--accent:{accent}">'
                     f'<div class="label">{it["label"]}</div>'
                     f'<div class="value">{it["value"]}</div>{sub}</div>')
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def split_rate(k, n, decimals=1):
    from dashboard_ext import wilson
    if n == 0:
        return "no data", ""
    lo, hi = wilson(k, n)
    return f"{100*k/n:.{decimals}f}%", f"CI {lo:.1f}\u2013{hi:.1f} \u00b7 n={n:,}"


def board_svg(board, played=None, best=None, size=250):
    """Board with the played move in blunder-red, the engine's move in slate."""
    arrows = []
    if played is not None:
        arrows.append(chess.svg.Arrow(played.from_square, played.to_square,
                                      color=SEVERITY["blunder"]))
    if best is not None:
        arrows.append(chess.svg.Arrow(best.from_square, best.to_square,
                                      color=SEVERITY["good"]))
    return chess.svg.board(board, arrows=arrows, size=size,
                           orientation=board.turn, coordinates=False)


def board_strip(st, figures):
    """figures: list of dicts with 'svg' and 'caption' (caption may contain <b>)."""
    html = "".join(f'<div class="board-fig">{f["svg"]}'
                   f'<div class="cap">{f["caption"]}</div></div>' for f in figures)
    st.markdown(f'<div class="boards">{html}</div>', unsafe_allow_html=True)


def severity_bar(st, series, x_label, y_label, order=None, higher_is_worse=True):
    """Bar chart via Altair -- st.bar_chart can't colour individual bars.

    Every bar is BAR_BASE except the single worst one, highlighted in
    SEVERITY["blunder"]. One red bar reads as "this is the problem"; every
    bar red carries no information at all.

    `higher_is_worse` picks which end is "worst": True for blunder rate /
    wp_loss (a high number is bad), False for best-move rate / win rate
    (a LOW number is bad). Get this backwards and the chart highlights your
    best category as if it were the problem -- pick per chart, not by default.

    `order` sorts the x-axis in that order (Altair handles this directly --
    no CategoricalIndex workaround needed here, unlike st.bar_chart).
    """
    import altair as alt

    data = series.rename("value").rename_axis(x_label).reset_index()
    worst_i = data["value"].idxmax() if higher_is_worse else data["value"].idxmin()
    data["is_worst"] = False
    data.loc[worst_i, "is_worst"] = True

    x_enc = alt.X(f"{x_label}:N", title=None, sort=order if order else "-y")
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=x_enc,
            y=alt.Y("value:Q", title=y_label),
            color=alt.condition(alt.datum.is_worst,
                                alt.value(SEVERITY["blunder"]), alt.value(BAR_BASE)),
            tooltip=[alt.Tooltip(f"{x_label}:N", title=x_label),
                     alt.Tooltip("value:Q", title=y_label, format=".1f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def trend_chart(st, df, metric_label, value_col, freq="W", color=None):
    """Weekly trend line over end_time. Returns the series."""
    import pandas as pd
    d = df.dropna(subset=["end_time"]).copy()
    d["period"] = pd.to_datetime(d.end_time).dt.to_period(freq).dt.to_timestamp()
    s = d.groupby("period")[value_col].mean()
    if d[value_col].dtype == bool or value_col in ("blunder", "won", "played_best"):
        s = s * 100
    st.line_chart(s.rename(metric_label),
                  color=color or BAR_BASE, height=280)
    return s
