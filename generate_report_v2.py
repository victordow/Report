#!/usr/bin/env python3
"""
Gera relatório HTML completo v2 — usando as colunas reais do hft_sim.

Colunas detections: detection_id, detected_at_ms, event_idx, event_title, category,
                    volume_bucket, time_bucket, direction, initial_sum_ask,
                    initial_sum_bid, initial_gross_edge, book_depth_bucket,
                    days_until_resolution, n_outcomes

Colunas latency:    detection_id, latency_target_ms, measured_at_ms,
                    actual_delay_ms, surviving_edge, book_available
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def find_latest_run(output_dir: Path) -> Path:
    runs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not runs:
        raise SystemExit(f"Nenhum run encontrado em {output_dir}")
    return runs[-1]


def load_all_parquets(run_dir: Path, prefix: str) -> pd.DataFrame:
    files = sorted(run_dir.glob(f"{prefix}_*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Erro lendo {f.name}: {e}", file=sys.stderr)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def pct_or_zero(numerator, denominator):
    return round(100.0 * numerator / denominator, 2) if denominator > 0 else 0


def build_html(run_dir: Path, detections: pd.DataFrame, latency: pd.DataFrame) -> str:

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_det = len(detections)
    total_lat = len(latency)

    # ========= EDGE STATS =========
    edge_col = "initial_gross_edge"
    edges = detections[edge_col].dropna() if edge_col in detections.columns else pd.Series([])

    edge_stats = {}
    if len(edges) > 0:
        edge_stats = {
            "mean": float(edges.mean()),
            "median": float(edges.median()),
            "p25": float(edges.quantile(0.25)),
            "p75": float(edges.quantile(0.75)),
            "p90": float(edges.quantile(0.90)),
            "p99": float(edges.quantile(0.99)),
            "max": float(edges.max()),
        }

    # faixas de edge bruto
    bands = {}
    if len(edges) > 0:
        bands = {
            "0-0.5%": int(((edges >= 0) & (edges < 0.005)).sum()),
            "0.5-1%": int(((edges >= 0.005) & (edges < 0.01)).sum()),
            "1-2%": int(((edges >= 0.01) & (edges < 0.02)).sum()),
            "2-3%": int(((edges >= 0.02) & (edges < 0.03)).sum()),
            "3-5%": int(((edges >= 0.03) & (edges < 0.05)).sum()),
            "5-10%": int(((edges >= 0.05) & (edges < 0.10)).sum()),
            ">10%": int((edges >= 0.10).sum()),
        }

    # histograma detalhado
    edge_hist = {}
    if len(edges) > 0:
        edges_pct = edges * 100
        edges_pct_clipped = edges_pct[(edges_pct >= 0) & (edges_pct <= 15)]
        if len(edges_pct_clipped) > 0:
            hist, bin_edges = np.histogram(edges_pct_clipped, bins=50)
            edge_hist = {
                "bins": [f"{bin_edges[i]:.2f}" for i in range(len(hist))],
                "counts": [int(c) for c in hist],
            }

    # ========= LATENCY SURVIVAL =========
    # latency: cada detecção tem várias rows, uma por latency_target_ms
    lat_summary = {}
    if len(latency) > 0 and "latency_target_ms" in latency.columns:
        for target_ms in sorted(latency["latency_target_ms"].unique()):
            sub = latency[latency["latency_target_ms"] == target_ms]
            total = len(sub)
            # sobrevive: surviving_edge > 0
            survived = int((sub["surviving_edge"] > 0).sum()) if "surviving_edge" in sub.columns else 0
            # sobrevive com edge > 0.5%
            survived_05 = int((sub["surviving_edge"] > 0.005).sum()) if "surviving_edge" in sub.columns else 0
            # sobrevive com edge > 1%
            survived_1 = int((sub["surviving_edge"] > 0.01).sum()) if "surviving_edge" in sub.columns else 0
            # sobrevive com edge > 2%
            survived_2 = int((sub["surviving_edge"] > 0.02).sum()) if "surviving_edge" in sub.columns else 0
            # sobrevive com edge > 3%
            survived_3 = int((sub["surviving_edge"] > 0.03).sum()) if "surviving_edge" in sub.columns else 0

            lat_summary[f"{int(target_ms)}ms"] = {
                "total": total,
                "surv_any": survived,
                "pct_any": pct_or_zero(survived, total),
                "surv_0.5": survived_05,
                "pct_0.5": pct_or_zero(survived_05, total),
                "surv_1": survived_1,
                "pct_1": pct_or_zero(survived_1, total),
                "surv_2": survived_2,
                "pct_2": pct_or_zero(survived_2, total),
                "surv_3": survived_3,
                "pct_3": pct_or_zero(survived_3, total),
            }

    # estatísticas de delay real
    delay_stats = {}
    if len(latency) > 0 and "actual_delay_ms" in latency.columns:
        delays = latency["actual_delay_ms"].dropna()
        if len(delays) > 0:
            delay_stats = {
                "median": float(delays.median()),
                "p90": float(delays.quantile(0.90)),
                "p99": float(delays.quantile(0.99)),
            }

    # ========= APÓS FEES =========
    # fees aplicadas sobre edge bruto: net = edge - fee
    # para cenários de 2%, 3%, 5%, 7%
    fee_scenarios = {}
    if len(edges) > 0:
        for fee_pct, label in [(0.02, "Fee 2%"), (0.03, "Fee 3%"), (0.05, "Fee 5%"), (0.07, "Fee 7%")]:
            net = edges - fee_pct
            fee_scenarios[label] = {
                "net_gt_0": int((net > 0).sum()),
                "net_gt_0.5": int((net > 0.005).sum()),
                "net_gt_1": int((net > 0.01).sum()),
                "net_gt_2": int((net > 0.02).sum()),
            }

    # ========= POR CATEGORIA =========
    cat_breakdown = {}
    if "category" in detections.columns:
        cat_counts = detections["category"].value_counts().head(15)
        for cat, count in cat_counts.items():
            sub_edges = detections[detections["category"] == cat][edge_col].dropna()
            cat_breakdown[str(cat)] = {
                "count": int(count),
                "median_edge": float(sub_edges.median()) if len(sub_edges) > 0 else 0,
                "p90_edge": float(sub_edges.quantile(0.90)) if len(sub_edges) > 0 else 0,
                "gt_1pct": int((sub_edges > 0.01).sum()) if len(sub_edges) > 0 else 0,
                "gt_3pct": int((sub_edges > 0.03).sum()) if len(sub_edges) > 0 else 0,
            }

    # ========= POR VOLUME BUCKET =========
    vol_breakdown = {}
    if "volume_bucket" in detections.columns:
        vol_counts = detections["volume_bucket"].value_counts()
        for vb, count in vol_counts.items():
            sub_edges = detections[detections["volume_bucket"] == vb][edge_col].dropna()
            vol_breakdown[str(vb)] = {
                "count": int(count),
                "median_edge": float(sub_edges.median()) if len(sub_edges) > 0 else 0,
                "gt_1pct": int((sub_edges > 0.01).sum()) if len(sub_edges) > 0 else 0,
                "gt_3pct": int((sub_edges > 0.03).sum()) if len(sub_edges) > 0 else 0,
            }

    # ========= POR BOOK DEPTH =========
    depth_breakdown = {}
    if "book_depth_bucket" in detections.columns:
        depth_counts = detections["book_depth_bucket"].value_counts()
        for db, count in depth_counts.items():
            sub_edges = detections[detections["book_depth_bucket"] == db][edge_col].dropna()
            depth_breakdown[str(db)] = {
                "count": int(count),
                "median_edge": float(sub_edges.median()) if len(sub_edges) > 0 else 0,
                "gt_1pct": int((sub_edges > 0.01).sum()) if len(sub_edges) > 0 else 0,
            }

    # ========= POR TIME BUCKET (tempo até resolução) =========
    time_breakdown = {}
    if "time_bucket" in detections.columns:
        time_counts = detections["time_bucket"].value_counts()
        for tb, count in time_counts.items():
            sub_edges = detections[detections["time_bucket"] == tb][edge_col].dropna()
            time_breakdown[str(tb)] = {
                "count": int(count),
                "median_edge": float(sub_edges.median()) if len(sub_edges) > 0 else 0,
                "gt_1pct": int((sub_edges > 0.01).sum()) if len(sub_edges) > 0 else 0,
            }

    # ========= TIMELINE =========
    timeline = {}
    if "detected_at_ms" in detections.columns:
        try:
            df = detections.copy()
            df["_ts"] = pd.to_datetime(df["detected_at_ms"], unit="ms", errors="coerce")
            df = df.dropna(subset=["_ts"])
            if len(df) > 0:
                df["_bucket"] = df["_ts"].dt.floor("5min")
                tl = df.groupby("_bucket").size()
                timeline = {
                    "x": [t.strftime("%H:%M") for t in tl.index],
                    "y": [int(v) for v in tl.values],
                }
        except Exception as e:
            print(f"Timeline erro: {e}", file=sys.stderr)

    # ========= LATENCY SURVIVAL BY EDGE THRESHOLD =========
    # Para cada threshold de edge inicial, calcula taxa de sobrevivência em 1s
    surv_by_initial_edge = {}
    if len(latency) > 0 and len(detections) > 0 and "surviving_edge" in latency.columns:
        # join latency com detections pra ter o initial_gross_edge
        try:
            merged = latency.merge(
                detections[["detection_id", "initial_gross_edge"]],
                on="detection_id",
                how="left"
            )
            # filtra só 1000ms
            lat1s = merged[merged["latency_target_ms"] == 1000]
            # bins de initial edge
            for lo, hi, label in [(0, 0.005, "0-0.5%"), (0.005, 0.01, "0.5-1%"),
                                   (0.01, 0.02, "1-2%"), (0.02, 0.03, "2-3%"),
                                   (0.03, 0.05, "3-5%"), (0.05, 1.0, ">5%")]:
                sub = lat1s[(lat1s["initial_gross_edge"] >= lo) & (lat1s["initial_gross_edge"] < hi)]
                total = len(sub)
                surv = int((sub["surviving_edge"] > 0).sum())
                surv_1pct = int((sub["surviving_edge"] > 0.01).sum())
                surv_2pct = int((sub["surviving_edge"] > 0.02).sum())
                surv_by_initial_edge[label] = {
                    "total": total,
                    "surv_any": surv,
                    "pct_any": pct_or_zero(surv, total),
                    "surv_1pct": surv_1pct,
                    "pct_1pct": pct_or_zero(surv_1pct, total),
                    "surv_2pct": surv_2pct,
                    "pct_2pct": pct_or_zero(surv_2pct, total),
                }
        except Exception as e:
            print(f"Merge erro: {e}", file=sys.stderr)

    # ========= MONTA HTML =========
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HFT Report — {run_dir.name}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0a0e1a; color: #e4e8f0; margin: 0; padding: 12px; max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #4ade80; border-bottom: 2px solid #4ade80; padding-bottom: 8px; font-size: 1.5em; }}
  h2 {{ color: #60a5fa; margin-top: 28px; border-bottom: 1px solid #334155; padding-bottom: 6px; font-size: 1.2em; }}
  h3 {{ color: #fbbf24; font-size: 1.05em; }}
  p {{ line-height: 1.5; }}
  .meta {{ color: #94a3b8; font-size: 0.85em; margin-bottom: 20px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin: 10px 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }}
  .kpi {{ background: #0f172a; padding: 10px; border-radius: 6px; border-left: 4px solid #4ade80; }}
  .kpi .label {{ color: #94a3b8; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi .value {{ font-size: 1.5em; font-weight: bold; color: #e4e8f0; margin-top: 4px; }}
  .kpi.warn {{ border-left-color: #fbbf24; }}
  .kpi.bad {{ border-left-color: #ef4444; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.9em; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #60a5fa; font-size: 0.85em; }}
  tr:hover td {{ background: #1e293b; }}
  .chart {{ min-height: 320px; margin: 12px 0; }}
  .verdict {{ background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%); border-left: 4px solid #fbbf24; padding: 14px; border-radius: 6px; margin: 14px 0; }}
  .good {{ color: #4ade80; }} .warn {{ color: #fbbf24; }} .bad {{ color: #ef4444; }}
  code {{ background: #0f172a; padding: 2px 5px; border-radius: 3px; font-family: Menlo, monospace; font-size: 0.88em; }}
  .nav {{ position: sticky; top: 0; background: #0a0e1a; padding: 8px 0; margin-bottom: 14px; border-bottom: 1px solid #334155; z-index: 100; overflow-x: auto; white-space: nowrap; }}
  .nav a {{ color: #60a5fa; margin-right: 14px; text-decoration: none; font-size: 0.85em; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>

<h1>🔬 HFT Simulation — Análise Completa</h1>
<div class="meta">
  Run: <code>{run_dir.name}</code> | Gerado: {gen_time}<br>
  <strong>{total_det:,}</strong> detecções | <strong>{total_lat:,}</strong> latency samples
</div>

<div class="nav">
  <a href="#kpi">KPIs</a>
  <a href="#edge">Edge</a>
  <a href="#lat">Latência</a>
  <a href="#fees">Após Fees</a>
  <a href="#cat">Categoria</a>
  <a href="#vol">Volume</a>
  <a href="#depth">Book Depth</a>
  <a href="#time">Time to Resolve</a>
  <a href="#timeline">Timeline</a>
  <a href="#verdict">Veredicto</a>
</div>

<h2 id="kpi">📊 KPIs de Alto Nível</h2>
<div class="card">
  <div class="grid">
    <div class="kpi"><div class="label">Total Detecções</div><div class="value">{total_det:,}</div></div>
    <div class="kpi"><div class="label">Detecções/hora</div><div class="value">{total_det // 2:,}</div></div>
    <div class="kpi"><div class="label">Detecções/min</div><div class="value">{total_det // 120:,}</div></div>
    <div class="kpi"><div class="label">Edge Mediano</div><div class="value">{edge_stats.get('median', 0)*100:.2f}%</div></div>
    <div class="kpi"><div class="label">Edge P90</div><div class="value">{edge_stats.get('p90', 0)*100:.2f}%</div></div>
    <div class="kpi"><div class="label">Edge Máximo</div><div class="value">{edge_stats.get('max', 0)*100:.2f}%</div></div>
  </div>
</div>

<h2 id="edge">📈 Distribuição de Edge Bruto</h2>
<p>Como o edge inicial se distribui. Lembre que <strong>edge bruto não é lucro</strong> — fees e slippage descontam.</p>
"""

    if bands:
        html += '<h3>Detecções por faixa</h3><table><tr><th>Faixa</th><th class="num">Detecções</th><th class="num">% total</th></tr>'
        for band, count in bands.items():
            pct = pct_or_zero(count, total_det)
            html += f'<tr><td>{band}</td><td class="num">{count:,}</td><td class="num">{pct}%</td></tr>'
        html += '</table>'

        bl = list(bands.keys())
        bc = list(bands.values())
        html += f"""
<div id="c-bands" class="chart"></div>
<script>
Plotly.newPlot('c-bands', [{{x: {json.dumps(bl)}, y: {json.dumps(bc)}, type: 'bar', marker: {{color: '#60a5fa'}}}}],
  {{title: 'Detecções por faixa de edge', paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a', font: {{color: '#e4e8f0'}}, xaxis: {{title: 'Edge'}}, yaxis: {{title: 'Detecções'}}}},
  {{responsive: true}});
</script>
"""

    if edge_hist:
        html += f"""
<h3>Histograma detalhado</h3>
<div id="c-hist" class="chart"></div>
<script>
Plotly.newPlot('c-hist', [{{x: {json.dumps(edge_hist["bins"])}, y: {json.dumps(edge_hist["counts"])}, type: 'bar', marker: {{color: '#4ade80'}}}}],
  {{title: 'Distribuição de edge (%)', paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a', font: {{color: '#e4e8f0'}}, xaxis: {{title: 'Edge (%)'}}, yaxis: {{title: 'Frequência'}}}},
  {{responsive: true}});
</script>
"""

    # LATÊNCIA
    html += '<h2 id="lat">⏱️ Sobrevivência por Latência</h2>'
    html += '<p>Das detecções com latency samples, quantas ainda tinham edge após X ms. Mostra sensibilidade da estratégia a delay.</p>'

    if delay_stats:
        html += f"""<div class="card"><div class="grid">
        <div class="kpi"><div class="label">Delay mediano real</div><div class="value">{delay_stats['median']:.0f}ms</div></div>
        <div class="kpi"><div class="label">Delay P90</div><div class="value">{delay_stats['p90']:.0f}ms</div></div>
        <div class="kpi"><div class="label">Delay P99</div><div class="value">{delay_stats['p99']:.0f}ms</div></div>
        </div></div>"""

    if lat_summary:
        html += '<table><tr><th>Latência</th><th class="num">Total</th><th class="num">Surv. qualquer</th><th class="num">Surv. >1%</th><th class="num">Surv. >2%</th><th class="num">Surv. >3%</th></tr>'
        for lat, d in lat_summary.items():
            html += f'<tr><td><strong>{lat}</strong></td><td class="num">{d["total"]:,}</td>'
            html += f'<td class="num">{d["pct_any"]}%</td><td class="num">{d["pct_1"]}%</td>'
            html += f'<td class="num">{d["pct_2"]}%</td><td class="num">{d["pct_3"]}%</td></tr>'
        html += '</table>'

        # chart multi-series
        lat_keys = list(lat_summary.keys())
        surv_any_pcts = [lat_summary[k]["pct_any"] for k in lat_keys]
        surv_1_pcts = [lat_summary[k]["pct_1"] for k in lat_keys]
        surv_2_pcts = [lat_summary[k]["pct_2"] for k in lat_keys]
        surv_3_pcts = [lat_summary[k]["pct_3"] for k in lat_keys]

        html += f"""
<div id="c-lat" class="chart"></div>
<script>
Plotly.newPlot('c-lat', [
  {{x: {json.dumps(lat_keys)}, y: {json.dumps(surv_any_pcts)}, name: 'Qualquer edge>0', type: 'bar', marker: {{color: '#4ade80'}}}},
  {{x: {json.dumps(lat_keys)}, y: {json.dumps(surv_1_pcts)}, name: 'Edge>1%', type: 'bar', marker: {{color: '#60a5fa'}}}},
  {{x: {json.dumps(lat_keys)}, y: {json.dumps(surv_2_pcts)}, name: 'Edge>2%', type: 'bar', marker: {{color: '#fbbf24'}}}},
  {{x: {json.dumps(lat_keys)}, y: {json.dumps(surv_3_pcts)}, name: 'Edge>3%', type: 'bar', marker: {{color: '#f87171'}}}}
], {{title: 'Taxa de sobrevivência por latência e threshold de edge', barmode: 'group', paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a', font: {{color: '#e4e8f0'}}, yaxis: {{title: '% sobrevivência', range: [0, 100]}}}},
  {{responsive: true}});
</script>
"""

    if surv_by_initial_edge:
        html += '<h3>Sobrevivência em 1s por faixa de edge inicial</h3>'
        html += '<p>Pergunta-chave: oportunidades com edge inicial maior sobrevivem mais? Se sim, a estratégia tem valor mesmo em latência alta.</p>'
        html += '<table><tr><th>Edge inicial</th><th class="num">Amostras</th><th class="num">Surv qualquer</th><th class="num">Surv >1%</th><th class="num">Surv >2%</th></tr>'
        for band, d in surv_by_initial_edge.items():
            html += f'<tr><td><strong>{band}</strong></td><td class="num">{d["total"]:,}</td>'
            html += f'<td class="num">{d["pct_any"]}%</td><td class="num">{d["pct_1pct"]}%</td><td class="num">{d["pct_2pct"]}%</td></tr>'
        html += '</table>'

        labels = list(surv_by_initial_edge.keys())
        pct_1 = [surv_by_initial_edge[k]["pct_1pct"] for k in labels]
        pct_2 = [surv_by_initial_edge[k]["pct_2pct"] for k in labels]
        html += f"""
<div id="c-surv-edge" class="chart"></div>
<script>
Plotly.newPlot('c-surv-edge', [
  {{x: {json.dumps(labels)}, y: {json.dumps(pct_1)}, name: 'Surv >1% em 1s', type: 'bar', marker: {{color: '#60a5fa'}}}},
  {{x: {json.dumps(labels)}, y: {json.dumps(pct_2)}, name: 'Surv >2% em 1s', type: 'bar', marker: {{color: '#fbbf24'}}}}
], {{title: 'Sobrevivência em 1s por edge inicial', barmode: 'group', paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a', font: {{color: '#e4e8f0'}}, yaxis: {{title: '% surv'}}}},
  {{responsive: true}});
</script>
"""

    # AFTER FEES
    html += '<h2 id="fees">💰 Após Fees</h2>'
    html += '<p>Fees Polymarket variam: ~2% típico, 7% em mercados crypto (taker). <strong>Este é o número que importa.</strong></p>'

    if fee_scenarios:
        html += '<table><tr><th>Cenário</th><th class="num">Net >0%</th><th class="num">Net >0.5%</th><th class="num">Net >1%</th><th class="num">Net >2%</th></tr>'
        for sc, d in fee_scenarios.items():
            html += f'<tr><td><strong>{sc}</strong></td>'
            html += f'<td class="num">{d["net_gt_0"]:,}</td><td class="num">{d["net_gt_0.5"]:,}</td>'
            html += f'<td class="num">{d["net_gt_1"]:,}</td><td class="num">{d["net_gt_2"]:,}</td></tr>'
        html += '</table>'

        scs = list(fee_scenarios.keys())
        html += f"""
<div id="c-fees" class="chart"></div>
<script>
Plotly.newPlot('c-fees', [
  {{x: {json.dumps(scs)}, y: {json.dumps([fee_scenarios[s]["net_gt_0"] for s in scs])}, name: 'Net >0', type: 'bar', marker: {{color: '#4ade80'}}}},
  {{x: {json.dumps(scs)}, y: {json.dumps([fee_scenarios[s]["net_gt_0.5"] for s in scs])}, name: 'Net >0.5%', type: 'bar', marker: {{color: '#60a5fa'}}}},
  {{x: {json.dumps(scs)}, y: {json.dumps([fee_scenarios[s]["net_gt_1"] for s in scs])}, name: 'Net >1%', type: 'bar', marker: {{color: '#fbbf24'}}}},
  {{x: {json.dumps(scs)}, y: {json.dumps([fee_scenarios[s]["net_gt_2"] for s in scs])}, name: 'Net >2%', type: 'bar', marker: {{color: '#f87171'}}}}
], {{title: 'Detecções líquidas por cenário de fee', barmode: 'group', paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a', font: {{color: '#e4e8f0'}}, yaxis: {{title: 'Detecções'}}}},
  {{responsive: true}});
</script>
"""

    # POR CATEGORIA
    if cat_breakdown:
        html += '<h2 id="cat">🏷️ Por Categoria</h2>'
        html += '<p>Quais categorias concentram mais oportunidades e qual a qualidade delas.</p>'
        html += '<table><tr><th>Categoria</th><th class="num">Detecções</th><th class="num">Edge mediano</th><th class="num">Edge P90</th><th class="num">>1%</th><th class="num">>3%</th></tr>'
        for cat, d in cat_breakdown.items():
            html += f'<tr><td>{cat}</td><td class="num">{d["count"]:,}</td>'
            html += f'<td class="num">{d["median_edge"]*100:.2f}%</td><td class="num">{d["p90_edge"]*100:.2f}%</td>'
            html += f'<td class="num">{d["gt_1pct"]:,}</td><td class="num">{d["gt_3pct"]:,}</td></tr>'
        html += '</table>'

        cats = list(cat_breakdown.keys())[:10]
        counts = [cat_breakdown[c]["count"] for c in cats]
        html += f"""
<div id="c-cat" class="chart"></div>
<script>
Plotly.newPlot('c-cat', [{{labels: {json.dumps(cats)}, values: {json.dumps(counts)}, type: 'pie', hole: 0.4}}],
  {{title: 'Distribuição por categoria', paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a', font: {{color: '#e4e8f0'}}}},
  {{responsive: true}});
</script>
"""

    # POR VOLUME
    if vol_breakdown:
        html += '<h2 id="vol">💵 Por Volume Bucket</h2>'
        html += '<table><tr><th>Volume</th><th class="num">Detecções</th><th class="num">Edge mediano</th><th class="num">>1%</th><th class="num">>3%</th></tr>'
        for vb, d in vol_breakdown.items():
            html += f'<tr><td>{vb}</td><td class="num">{d["count"]:,}</td>'
            html += f'<td class="num">{d["median_edge"]*100:.2f}%</td>'
            html += f'<td class="num">{d["gt_1pct"]:,}</td><td class="num">{d["gt_3pct"]:,}</td></tr>'
        html += '</table>'

    # POR BOOK DEPTH
    if depth_breakdown:
        html += '<h2 id="depth">📚 Por Book Depth</h2>'
        html += '<p>Profundidade do book na hora da detecção. Book raso = pouca liquidez, risco maior de slippage.</p>'
        html += '<table><tr><th>Depth</th><th class="num">Detecções</th><th class="num">Edge mediano</th><th class="num">>1%</th></tr>'
        for db, d in depth_breakdown.items():
            html += f'<tr><td>{db}</td><td class="num">{d["count"]:,}</td>'
            html += f'<td class="num">{d["median_edge"]*100:.2f}%</td>'
            html += f'<td class="num">{d["gt_1pct"]:,}</td></tr>'
        html += '</table>'

    # POR TIME BUCKET
    if time_breakdown:
        html += '<h2 id="time">⏳ Por Tempo até Resolução</h2>'
        html += '<table><tr><th>Time bucket</th><th class="num">Detecções</th><th class="num">Edge mediano</th><th class="num">>1%</th></tr>'
        for tb, d in time_breakdown.items():
            html += f'<tr><td>{tb}</td><td class="num">{d["count"]:,}</td>'
            html += f'<td class="num">{d["median_edge"]*100:.2f}%</td>'
            html += f'<td class="num">{d["gt_1pct"]:,}</td></tr>'
        html += '</table>'

    # TIMELINE
    if timeline:
        html += '<h2 id="timeline">📅 Timeline</h2>'
        html += f"""
<div id="c-tl" class="chart"></div>
<script>
Plotly.newPlot('c-tl', [{{x: {json.dumps(timeline["x"])}, y: {json.dumps(timeline["y"])}, type: 'scatter', mode: 'lines+markers', line: {{color: '#4ade80', width: 2}}, fill: 'tozeroy', fillcolor: 'rgba(74, 222, 128, 0.1)'}}],
  {{title: 'Detecções por 5min', paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a', font: {{color: '#e4e8f0'}}, xaxis: {{title: 'Hora'}}, yaxis: {{title: 'Detecções'}}}},
  {{responsive: true}});
</script>
"""

    # VEREDICTO
    html += '<h2 id="verdict">🎯 Veredicto Honesto</h2>'

    if fee_scenarios:
        fee_2 = fee_scenarios.get("Fee 2%", {})
        fee_7 = fee_scenarios.get("Fee 7%", {})

        html += '<div class="card">'
        html += '<h3>Com fee 2% (mercados Polymarket típicos — não crypto)</h3>'
        html += f'<ul>'
        html += f'<li><strong>{fee_2.get("net_gt_0", 0):,}</strong> detecções net &gt; 0 ({fee_2.get("net_gt_0", 0) // 2:,}/hora)</li>'
        html += f'<li><strong>{fee_2.get("net_gt_1", 0):,}</strong> detecções net &gt; 1% ({fee_2.get("net_gt_1", 0) // 2:,}/hora)</li>'
        html += f'<li><strong>{fee_2.get("net_gt_2", 0):,}</strong> detecções net &gt; 2%</li>'
        html += f'</ul>'

        html += '<h3>Com fee 7% (mercados crypto com taker fee ativo)</h3>'
        html += f'<ul>'
        html += f'<li><strong>{fee_7.get("net_gt_0", 0):,}</strong> detecções net &gt; 0</li>'
        html += f'<li><strong>{fee_7.get("net_gt_1", 0):,}</strong> detecções net &gt; 1%</li>'
        html += f'</ul>'
        html += '</div>'

        per_hour_2_gt1 = fee_2.get("net_gt_1", 0) // 2
        per_hour_7_gt1 = fee_7.get("net_gt_1", 0) // 2

        if per_hour_2_gt1 > 100:
            verdict = '<p class="good"><strong>✅ Sinal forte.</strong> Volume substancial de oportunidades com edge líquido relevante em mercados de fee 2%. HFT em Python parece viável para mercados típicos (sports, politics, culture).</p>'
        elif per_hour_2_gt1 > 20:
            verdict = '<p class="warn"><strong>⚠️ Sinal moderado.</strong> Oportunidades existem mas requerem boa infraestrutura e talvez seletividade de mercados.</p>'
        else:
            verdict = '<p class="bad"><strong>❌ Sinal fraco.</strong> Poucas oportunidades sobrevivem fees típicas. Estratégia questionável em escala.</p>'

        html += f'<div class="verdict">{verdict}</div>'

        if per_hour_7_gt1 < 10:
            html += '<div class="verdict"><p class="bad"><strong>❌ Crypto markets com fee 7% são inviáveis.</strong> O edge bruto quase nunca cobre o taker fee. Evitar este tipo de mercado até fees caírem.</p></div>'

    html += """
<h3>⚠️ O que este teste NÃO mediu</h3>
<div class="card">
<ul>
  <li><strong>Execução real:</strong> o book é passivo. Quando sua ordem chega, o preço se move.</li>
  <li><strong>Competição de bots:</strong> outros bots com latência menor capturam primeiro.</li>
  <li><strong>Fill probability:</strong> nem toda ordem com edge positivo é executada.</li>
  <li><strong>Gas em NegRisk:</strong> múltiplas legs têm custos on-chain.</li>
  <li><strong>Rebalancing real:</strong> com $2-3k, capital fica preso até settle ou reversal.</li>
</ul>
<p>Próximo passo honesto: <strong>paper trading</strong> comparando detecções com execuções que de fato teriam fillado.</p>
</div>

</body>
</html>
"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Nome do run")
    ap.add_argument("--output-dir", default=str(Path.home() / "hft" / "output"))
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    run_dir = output_dir / args.run if args.run else find_latest_run(output_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run não encontrado: {run_dir}")

    print(f"Analisando: {run_dir}")
    detections = load_all_parquets(run_dir, "detections")
    latency = load_all_parquets(run_dir, "latency")
    print(f"Detecções: {len(detections):,}")
    print(f"Latency samples: {len(latency):,}")

    html = build_html(run_dir, detections, latency)
    out = run_dir / "report_v2.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n✅ Relatório: {out}")
    print(f"Tamanho: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
