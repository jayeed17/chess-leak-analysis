"""Theme + KPI cards for app.py.

    from theme import inject_theme, kpi_row, QUALITY
    inject_theme(st)
    kpi_row(st, [...])

Palette comes from the chess.com board: light square #EEEED2 is the page, dark
square #769656 is structure. KPI accents are the move-quality colors chess
interfaces already use, so the colour tells you which class the number belongs to
rather than just varying.
"""

# board
PAPER = "#EEEED2"      # light square — page
PANEL = "#E4E4C4"      # light square, one step down — cards
RULE = "#C3C9A5"       # hairline
DARK = "#769656"       # dark square — structure, headers
INK = "#22261A"        # board ink — body text
MUTED = "#5C6350"      # secondary text

# move-quality accents
QUALITY = {
    "brilliant": "#1BACA6",
    "good": "#769656",
    "inaccuracy": "#C8952B",
    "mistake": "#A93226",
    "neutral": "#5C6350",
}

_CSS = f"""
<style>
  html, body, [class*="css"], .stApp,
  [data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
  .stDataFrame, .stSelectbox, .stMultiSelect, button, input, label {{
    font-family: "Times New Roman", Times, serif !important;
  }}

  .stApp {{ background: {PAPER}; color: {INK}; }}
  [data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {RULE};
  }}

  h1, h2, h3 {{
    color: {INK};
    font-weight: 700;
    letter-spacing: -0.01em;
    line-height: 1.15;
  }}
  h1 {{ font-size: 2.6rem; margin-bottom: 0.1em; }}
  h2 {{ font-size: 1.6rem; }}
  h3 {{ font-size: 1.2rem; }}

  /* serif body wants more leading and a shorter measure */
  [data-testid="stMarkdownContainer"] p {{
    line-height: 1.65;
    max-width: 74ch;
    color: {INK};
  }}
  [data-testid="stCaptionContainer"] p {{
    color: {MUTED};
    font-size: 0.92rem;
    line-height: 1.55;
    max-width: 74ch;
  }}

  .stTabs [data-baseweb="tab-list"] {{
    gap: 1.75rem;
    border-bottom: 1px solid {RULE};
  }}
  .stTabs [data-baseweb="tab"] {{
    font-size: 1.05rem;
    color: {MUTED};
    padding: 0.4rem 0;
  }}
  .stTabs [aria-selected="true"] {{
    color: {INK};
    border-bottom: 2px solid {DARK};
  }}

  /* KPI card: quality colour on the left edge, number large, interval quiet */
  .kpi-row {{ display: flex; gap: 0.9rem; flex-wrap: wrap; margin: 0.5rem 0 1.4rem; }}
  .kpi {{
    flex: 1 1 190px;
    background: {PANEL};
    border: 1px solid {RULE};
    border-left: 4px solid var(--accent);
    padding: 0.85rem 1rem 0.9rem;
  }}
  .kpi .label {{
    font-size: 0.95rem;
    color: {MUTED};
    margin-bottom: 0.15rem;
  }}
  .kpi .value {{
    font-size: 2.15rem;
    line-height: 1.05;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }}
  .kpi .sub {{
    font-size: 0.85rem;
    color: {MUTED};
    font-variant-numeric: tabular-nums;
    margin-top: 0.2rem;
  }}
  .kpi.thin {{ border-left-style: dashed; }}
  .kpi.thin .sub::after {{ content: " · thin sample"; color: {QUALITY['mistake']}; }}

  .stDataFrame {{ border: 1px solid {RULE}; }}

  @media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
  }}
</style>
"""


def inject_theme(st):
    st.markdown(_CSS, unsafe_allow_html=True)


def kpi_row(st, items):
    """items: list of dicts with keys
         label   - sentence-case name
         value   - the headline number, already formatted ('6.9%', '488')
         sub     - optional interval / sample line ('CI 5.8-8.2 · n=2,239')
         quality - key into QUALITY, picks the accent
         thin    - optional bool, dashes the edge and flags the sample
    """
    cards = []
    for it in items:
        accent = QUALITY.get(it.get("quality", "neutral"), QUALITY["neutral"])
        cls = "kpi thin" if it.get("thin") else "kpi"
        sub = f'<div class="sub">{it["sub"]}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="{cls}" style="--accent:{accent}">'
            f'<div class="label">{it["label"]}</div>'
            f'<div class="value">{it["value"]}</div>{sub}</div>'
        )
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def split_rate(k, n, decimals=1):
    """rate_ci() returns one long string, which is what breaks st.metric.
    This splits it into (value, sub) for a KPI card."""
    from dashboard_ext import wilson
    if n == 0:
        return "no data", ""
    lo, hi = wilson(k, n)
    return f"{100*k/n:.{decimals}f}%", f"CI {lo:.1f}–{hi:.1f} · n={n:,}"
