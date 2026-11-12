#!/usr/bin/env python3
"""Render a league standings.json as a standalone standings.html.

Interim companion to ollama_league.py's native standings.html: seasons
already running under an older league build do not write the HTML
page, so this converter runs where their output lands (the Vostro
share) and refreshes standings.html from the synced standings.json.
Fully self-contained on purpose -- the share host needs nothing but
Python's standard library. Once every league runs a build with native
HTML output, this tool has no job left.

Usage: standings_html.py DIR [DIR ...]
Each DIR holds a standings.json; standings.html is written beside it.
"""

import html
import json
import math
import os
import sys

MEDIAN = {"input": 0.435, "output": 2.10}

_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
  sans-serif; margin: 2em auto; max-width: 920px; padding: 0 1em;
  line-height: 1.45; color: #1a1a1a; background: #fff; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 0.35em 0.6em;
  text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {
  text-align: left; }
thead th { background: #f0f0f0; }
tbody tr:nth-child(even) { background: #fafafa; }
.meta { color: #555; }
.winner { font-weight: bold; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #16181d; }
  th, td { border-color: #444; }
  thead th { background: #24262c; }
  tbody tr:nth-child(even) { background: #1c1e24; }
  .meta { color: #9a9a9a; } }
"""


def cloud_cost(tokens_in, tokens_out):
    return (tokens_in * MEDIAN["input"]
            + tokens_out * MEDIAN["output"]) / 1_000_000


def format_cost(amount):
    """Mirror ollama_reporting.format_cost (kept in step by test)."""
    if amount <= 0:
        return "$0"
    if amount >= 1:
        return f"${amount:,.2f}"
    decimals = 2 - math.floor(math.log10(amount))
    return f"${amount:.{decimals}f}"


def sorted_table(state):
    """Mirror ollama_league.sorted_table."""
    return sorted(state["table"].items(),
                  key=lambda kv: (-kv[1]["points"], -kv[1]["won"],
                                  kv[1]["tokens_in"] + kv[1]["tokens_out"],
                                  kv[0]))


def render_html(state):
    esc = html.escape
    name = state.get("league") or "League"
    parts = ["<!DOCTYPE html>", '<html lang="en">', "<head>",
             '<meta charset="utf-8">',
             f"<title>{esc(name)} — standings</title>",
             f"<style>{_CSS}</style>", "</head>", "<body>",
             f"<h1>{esc(name)} — standings</h1>",
             f'<p class="meta">Updated: {esc(state.get("updated") or "-")}'
             f"<br>Cloud-equivalent cost at median rates "
             f"(${MEDIAN['input']} in / ${MEDIAN['output']} out per 1M "
             "tokens).</p>",
             "<table>", "<thead><tr><th>Pos</th><th>Model</th><th>P</th>"
             "<th>W</th><th>D</th><th>L</th><th>Pts</th><th>Tokens in</th>"
             "<th>Tokens out</th><th>Cloud-equiv</th></tr></thead>",
             "<tbody>"]
    for pos, (model, row) in enumerate(sorted_table(state), 1):
        cells = [str(pos), esc(model), str(row["played"]), str(row["won"]),
                 str(row["drawn"]), str(row["lost"]), str(row["points"]),
                 f"{row['tokens_in']:,}", f"{row['tokens_out']:,}",
                 format_cost(cloud_cost(row["tokens_in"],
                                        row["tokens_out"]))]
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells)
                     + "</tr>")
    parts += ["</tbody>", "</table>"]
    played = [m for m in state["matches"] if m.get("status") == "played"]
    parts.append(f"<h2>Matches ({len(played)} played)</h2>")
    parts.append("<ul>")
    for m in reversed(played[-25:]):
        outcome = m["winner"] if m.get("winner") else "draw"
        parts.append(f"<li>{esc(m['date'])} · {esc(m['scenario'])} · "
                     f"{esc(m['a'])} vs {esc(m['b'])} → "
                     f'<span class="winner">{esc(outcome)}</span> '
                     f"(judge: {esc(m['judge'])})</li>")
    parts.append("</ul>")
    parts += ["</body>", "</html>"]
    return "\n".join(parts) + "\n"


def convert(directory):
    src = os.path.join(directory, "standings.json")
    with open(src, encoding="utf-8") as fh:
        state = json.load(fh)
    dst = os.path.join(directory, "standings.html")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(render_html(state))
    return dst


def main(argv=None):
    dirs = (argv if argv is not None else sys.argv[1:])
    if not dirs:
        print(__doc__)
        return 2
    for directory in dirs:
        print(convert(directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
