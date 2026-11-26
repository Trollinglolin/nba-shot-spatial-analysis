"""Interactive Plotly shot charts: per-player profiles and a comparative dashboard."""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import constants as C
import features as F
import model as M

INK, MUTED, LINE = "#1f2933", "#6b7785", "#9aa5b1"
# Diverging blue (cold) <-> neutral gray <-> orange-red (hot); sequential single-hue for frequency
DIVERGING = [[0, "#2166ac"], [0.5, "#e6e8eb"], [1, "#d6532f"]]
SEQ = [[0, "#f3f6fb"], [1, "#1f4e8c"]]
PLAYER_C, LEAGUE_C = "#1f4e8c", "#9aa5b1"


def _arc(cx, cy, r, a0, a1, n=60):
    t = np.radians(np.linspace(a0, a1, n))
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _path(xs, ys):
    return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def court_shapes(xref="x", yref="y", color=LINE):
    """Half-court line drawing as plotly path/rect shapes (tenths of feet)."""
    kw = dict(xref=xref, yref=yref, line=dict(color=color, width=1.5), layer="above")
    s = [
        dict(type="rect", x0=-250, y0=C.BASELINE_Y, x1=250, y1=C.HALF_COURT_Y, **kw),
        dict(type="rect", x0=-C.PAINT_HALF_WIDTH, y0=C.BASELINE_Y, x1=C.PAINT_HALF_WIDTH, y1=C.FREE_THROW_Y, **kw),
        dict(type="rect", x0=-C.LANE_INNER_HALF_WIDTH, y0=C.BASELINE_Y, x1=C.LANE_INNER_HALF_WIDTH, y1=C.FREE_THROW_Y, **kw),
        dict(type="line", x0=-30, y0=C.BACKBOARD_Y, x1=30, y1=C.BACKBOARD_Y, **kw),
    ]
    arcs = [
        _arc(0, 0, C.RIM_RADIUS, 0, 360),                                   # rim
        _arc(0, 0, C.RESTRICTED_RADIUS, 0, 180),                            # restricted area
        _arc(0, C.FREE_THROW_Y, C.FREE_THROW_RADIUS, 0, 180),               # FT circle (top)
        _arc(0, C.HALF_COURT_Y, 60, 180, 360),                              # center circle
    ]
    ang = np.degrees(np.arccos(C.THREE_PT_CORNER_X / C.THREE_PT_ARC_RADIUS))
    ax, ay = _arc(0, 0, C.THREE_PT_ARC_RADIUS, ang, 180 - ang, 120)         # 3PT arc
    arcs.append((np.r_[ax[0] * 0 + C.THREE_PT_CORNER_X, ax, -C.THREE_PT_CORNER_X],
                 np.r_[C.BASELINE_Y, ay, C.BASELINE_Y]))
    for xs, ys in arcs:
        s.append(dict(type="path", path=_path(xs, ys), **kw))
    return s


def _bin_edges():
    xe = np.arange(C.GRID_X_RANGE[0], C.GRID_X_RANGE[1] + 1, C.GRID_BIN)
    ye = np.arange(C.GRID_Y_RANGE[0], C.GRID_Y_RANGE[1] + 1, C.GRID_BIN)
    return xe, ye


def grid_stats(df: pd.DataFrame):
    """Attempts and makes per bin. Returns (xc, yc, attempts, makes)."""
    xe, ye = _bin_edges()
    att, _, _ = np.histogram2d(df["loc_y"], df["loc_x"], bins=[ye, xe])
    mk, _, _ = np.histogram2d(df.loc[df[F.TARGET] == 1, "loc_y"], df.loc[df[F.TARGET] == 1, "loc_x"], bins=[ye, xe])
    return (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, att, mk


def _style_court(fig, row=None, col=None, title=None):
    kw = dict(row=row, col=col) if row else {}
    fig.update_xaxes(range=[-255, 255], visible=False, **kw)
    fig.update_yaxes(range=[-57, 425], visible=False, scaleanchor=None, **kw)


def _heat_traces(player_df, league_df, loc_model, season, min_att):
    """Efficiency-vs-league heatmap trace + league xFG contour trace."""
    xc, yc, pa, pm = grid_stats(player_df)
    _, _, la, lm = grid_stats(league_df)
    with np.errstate(invalid="ignore", divide="ignore"):
        pfg, lfg = pm / pa, lm / la
    diff = np.where(pa >= min_att, (pfg - lfg) * 100, np.nan)
    hover = np.where(pa >= min_att,
                     [[f"{int(pm[i, j])}/{int(pa[i, j])} = {pfg[i, j]:.0%}<br>league {lfg[i, j]:.0%}"
                       if pa[i, j] else "" for j in range(len(xc))] for i in range(len(yc))], "")
    heat = go.Heatmap(x=xc, y=yc, z=diff, text=hover, hoverinfo="text", colorscale=DIVERGING,
                      zmid=0, zmin=-25, zmax=25, xgap=2, ygap=2,
                      colorbar=dict(title="FG% vs league<br>(pct pts)", thickness=10, len=0.6),
                      showscale=True)
    gx, gy = np.arange(-245, 246, 10.0), np.arange(-50, 426, 10.0)
    cont = go.Contour(x=gx, y=gy, z=M.predict_grid(loc_model, gx, gy, season), showscale=False,
                      colorscale=[[0, MUTED], [1, MUTED]], contours=dict(coloring="lines", start=0.30, end=0.65, size=0.05,
                                    showlabels=True, labelfont=dict(size=9, color=MUTED)),
                      line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip", name="league xFG%")
    return heat, cont


def side_labels(fig, **kw):
    """Label court sides from the shooter's view (verified: nba_api LOC_X < 0 is `Left Side(L)`)."""
    for x, anchor, text in ((-248, "left", "&#9664; shooter's left"), (248, "right", "shooter's right &#9654;")):
        fig.add_annotation(x=x, y=-57, text=text, showarrow=False, xanchor=anchor, yanchor="top",
                           font=dict(size=10, color=MUTED), **kw)


def player_profile(player_df, league_df, models, name, min_att=3):
    """Single-player shot chart: FG% vs league by 4-ft bin with league xFG% contours."""
    min_att = max(min_att, round(len(player_df) / 500))      # busier players get a stricter per-cell minimum
    heat, cont = _heat_traces(player_df, league_df, models["loc"], models["latest_season"], min_att)
    fig = go.Figure([heat, cont])
    fig.update_layout(shapes=court_shapes())
    fig.update_xaxes(range=[-255, 255], visible=False)
    fig.update_yaxes(range=[-75, 425], visible=False, scaleanchor="x", scaleratio=1)
    side_labels(fig)
    n = len(player_df)
    fg = player_df[F.TARGET].mean()
    fig.update_layout(
        title=dict(text=f"{html.escape(name)} - {n:,} FGA, {fg:.1%} FG<br><sup>Color: FG% minus league FG% per 4-ft cell (blank if < {min_att} FGA)<br>"
                        f"Dotted lines: league-average xFG% by location ({models["latest_season"]})</sup>", x=0.02),
        width=640, height=600, margin=dict(l=10, r=10, t=100, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK))
    return fig


def zone_summary(df, xfg):
    d = df.assign(xfg=xfg)
    t = d.groupby("zone").agg(fga=(F.TARGET, "size"), fg=(F.TARGET, "mean"), xfg=("xfg", "mean"))
    t = t.reindex(C.ZONES)
    t["freq"] = t["fga"] / t["fga"].sum()
    return t


def comparison_dashboard(player_df, league_df, models, name):
    """2x2: shot chart | frequency vs league | distance distribution | zone FG% vs xFG%."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "FG% vs league by location", "Shot frequency by zone", "Shot distance distribution", "FG% by zone: actual vs expected"),
        vertical_spacing=0.14, horizontal_spacing=0.1)
    model = models["full"]
    heat, cont = _heat_traces(player_df, league_df, models["loc"], models["latest_season"], 3)
    fig.add_trace(heat, row=1, col=1)
    fig.add_trace(cont, row=1, col=1)
    zp = zone_summary(player_df, M.predict_xfg(model, player_df))
    zl = zone_summary(league_df, M.predict_xfg(model, league_df))
    # frequency bars (player vs league)
    fig.add_trace(go.Bar(x=C.ZONES, y=zp["freq"], name=name, marker_color=PLAYER_C, legendgroup="p",
                         hovertemplate="%{x}<br>%{y:.1%} of FGA<extra>" + html.escape(name) + "</extra>"), row=1, col=2)
    fig.add_trace(go.Bar(x=C.ZONES, y=zl["freq"], name="League", marker_color=LEAGUE_C, legendgroup="l",
                         hovertemplate="%{x}<br>%{y:.1%} of FGA<extra>League</extra>"), row=1, col=2)
    # distance distribution: binned shares (precomputed so the HTML does not embed every shot)
    edges = np.arange(0, 42, 2)
    centers = edges[:-1] + 1
    for d, nm, c, g in ((player_df, name, PLAYER_C, "p"), (league_df, "League", LEAGUE_C, "l")):
        share = np.histogram(d["distance_ft"], bins=edges)[0] / max(len(d), 1)
        fig.add_trace(go.Bar(x=centers, y=share, width=1.8, name=nm, marker_color=c,
                             opacity=0.9 if g == "p" else 0.75, legendgroup=g, showlegend=False,
                             hovertemplate="%{x:.0f} ft (&plusmn;1)<br>%{y:.1%}<extra>" + html.escape(nm) + "</extra>"),
                      row=2, col=1)
    # actual vs expected by zone
    fig.add_trace(go.Bar(x=C.ZONES, y=zp["fg"], name="Actual FG%", marker_color=PLAYER_C, legendgroup="a", showlegend=True,
                         hovertemplate="%{x}<br>actual %{y:.1%}<extra></extra>"), row=2, col=2)
    fig.add_trace(go.Bar(x=C.ZONES, y=zp["xfg"], name="xFG% (model)", marker=dict(color="white", line=dict(color=PLAYER_C, width=1.5), pattern=dict(shape="/", fgcolor=PLAYER_C)), legendgroup="x",
                         hovertemplate="%{x}<br>expected %{y:.1%}<extra></extra>"), row=2, col=2)
    for s in court_shapes("x", "y"):
        fig.add_shape(s)
    fig.update_xaxes(range=[-255, 255], visible=False, row=1, col=1)
    fig.update_yaxes(range=[-75, 425], visible=False, scaleanchor="x", row=1, col=1)
    side_labels(fig, xref="x", yref="y")
    fig.update_yaxes(tickformat=".0%", gridcolor="#eceff2", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", gridcolor="#eceff2", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", gridcolor="#eceff2", range=[0, 0.8], row=2, col=2)
    fig.update_xaxes(title_text="Shot distance (ft)", row=2, col=1)
    fig.update_xaxes(tickangle=-25, tickfont=dict(size=10), row=1, col=2)
    fig.update_xaxes(tickangle=-25, tickfont=dict(size=10), row=2, col=2)
    fp, fl = player_df[F.TARGET].mean(), league_df[F.TARGET].mean()
    xp = M.predict_xfg(model, player_df).mean()
    fig.update_layout(
        title=dict(text=f"{html.escape(name)} vs league: {len(player_df):,} FGA - FG {fp:.1%}, xFG {xp:.1%}, league FG {fl:.1%}", x=0.02),
        barmode="group", height=980, width=1100, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=INK), legend=dict(orientation="h", y=-0.06), margin=dict(l=40, r=20, t=90, b=60))
    fig.update_layout(barmode="group")
    return fig


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def export_all(test_df, models, out_dir=C.PATHS["charts"], featured=None, min_attempts=C.MIN_ATTEMPTS_PLAYER, max_players=150):
    """Write one profile HTML per player, a dashboard for `featured`, and an index page."""
    out = Path(out_dir)
    (out / "players").mkdir(parents=True, exist_ok=True)
    counts = test_df.groupby("player_name").size()
    players = counts[counts >= min_attempts].sort_values(ascending=False).head(max_players)
    featured = featured or players.index[0]
    cfg = dict(include_plotlyjs="cdn", config={"displaylogo": False})
    for name in players.index:
        player_profile(test_df[test_df.player_name == name], test_df, models, name).write_html(
            out / "players" / f"{slug(name)}.html", **cfg)
    comparison_dashboard(test_df[test_df.player_name == featured], test_df, models, featured).write_html(
        out / "dashboard.html", **cfg)
    rows = "\n".join(f'<li><a href="players/{slug(n)}.html">{html.escape(n)}</a> <span>{c:,} FGA</span></li>'
                     for n, c in players.items())
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Shot charts</title><style>body{font:16px system-ui;max-width:720px;margin:2rem auto;padding:0 16px;color:#1f2933}"
        "li{margin:.3rem 0}span{color:#6b7785;font-size:.85em;margin-left:.5rem}a{color:#1f4e8c}</style>"
        f"<h1>Player shot charts</h1><p><a href='dashboard.html'>Comparative dashboard: {html.escape(featured)}</a></p>"
        f"<ul>{rows}</ul>")
    return len(players), featured
