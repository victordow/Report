#!/usr/bin/env python3
"""
Gera relatório HTML completo com gráficos interativos a partir dos parquets do hft_sim.

Uso:
    python3 generate_report.py                              # auto-detecta último run
    python3 generate_report.py --run run_2026-04-20_00h19   # especifica run
"""

from __future__ import annotations
import argparse
import json
import os
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


def safe_stats(series: pd.Series) -> dict:
    """Estatísticas básicas, lidando com NaN."""
    s = series.dropna()
    if len(s) == 0:
        return {"count": 0, "mean": 0, "median": 0, "p25": 0, "p75": 0, "p90": 0, "p99": 0, "min": 0, "max": 0}
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def compute_edge_bands(detections: pd.DataFrame, edge_col: str = "edge") -> dict:
    """Quantas detecções caem em cada faixa de edge bruto."""
    if edge_col not in detections.columns or len(detections) == 0:
        return {}
    bands = {
        "0-0.5%": ((detections[edge_col] >= 0) & (detections[edge_col] < 0.005)).sum(),
        "0.5-1%": ((detections[edge_col] >= 0.005) & (detections[edge_col] < 0.01)).sum(),
        "1-2%": ((detections[edge_col] >= 0.01) & (detections[edge_col] < 0.02)).sum(),
        "2-3%": ((detections[edge_col] >= 0.02) & (detections[edge_col] < 0.03)).sum(),
        "3-5%": ((detections[edge_col] >= 0.03) & (detections[edge_col] < 0.05)).sum(),
        "5-10%": ((detections[edge_col] >= 0.05) & (detections[edge_col] < 0.10)).sum(),
        ">10%": (detections[edge_col] >= 0.10).sum(),
    }
    return {k: int(v) for k, v in bands.items()}


def compute_net_after_fees(detections: pd.DataFrame, fee: float, edge_col: str = "edge") -> dict:
    """Quantas detecções tem edge LÍQUIDO > 0, > 0.5%, > 1%, > 2% após descontar fee."""
    if edge_col not in detections.columns or len(detections) == 0:
        return {}
    net = detections[edge_col] - fee
    return {
        "total": int(len(detections)),
        "net_positive": int((net > 0).sum()),
        "net_gt_0.5%": int((net > 0.005).sum()),
        "net_gt_1%": int((net > 0.01).sum()),
        "net_gt_2%": int((net > 0.02).sum()),
    }


def compute_latency_survival(latency_df: pd.DataFrame) -> dict:
    """Taxa de sobrevivência por faixa de latência + threshold de edge."""
    if len(latency_df) == 0:
        return {}
    result = {}
    for lat_col_name, lat_label in [("survived_300ms", "300ms"),
                                      ("survived_1000ms", "1s"),
                                      ("survived_2000ms", "2s"),
                                      ("survived_5000ms", "5s")]:
        if lat_col_name not in latency_df.columns:
            continue
        total = len(latency_df)
        survived = int(latency_df[lat_col_name].sum())
        result[lat_label] = {
            "total": total,
            "survived": survived,
            "pct": round(100.0 * survived / total, 2) if total > 0 else 0,
        }
    return result


def build_html(run_dir: Path, detections: pd.DataFrame, latency: pd.DataFrame) -> str:
    """Monta o HTML com gráficos Plotly."""

    # --- stats básicos ---
    total_detections = len(detections)
    total_latency = len(latency)

    # descobrir colunas reais
    edge_col = None
    for cand in ["edge", "sum_yes_deviation", "deviation", "gross_edge"]:
        if cand in detections.columns:
            edge_col = cand
            break

    # timestamps
    ts_col = None
    for cand in ["timestamp", "ts", "time"]:
        if cand in detections.columns:
            ts_col = cand
            break

    # categorias
    cat_col = None
    for cand in ["category", "topic", "tag"]:
        if cand in detections.columns:
            cat_col = cand
            break

    # stats de edge
    edge_stats = safe_stats(detections[edge_col]) if edge_col else {}

    # bands
    bands = compute_edge_bands(detections, edge_col) if edge_col else {}

    # após fees
    fee_scenarios = {
        "Fee 2% (Poly típico)": compute_net_after_fees(detections, 0.02, edge_col),
        "Fee 3%": compute_net_after_fees(detections, 0.03, edge_col),
        "Fee 5%": compute_net_after_fees(detections, 0.05, edge_col),
        "Fee 7% (Crypto c/ taker fee)": compute_net_after_fees(detections, 0.07, edge_col),
    } if edge_col else {}

    # latência
    lat_stats = compute_latency_survival(latency)

    # categoria
    cat_counts = {}
    if cat_col and cat_col in detections.columns:
        cat_counts = detections[cat_col].value_counts().head(15).to_dict()
        cat_counts = {str(k): int(v) for k, v in cat_counts.items()}

    # timeline: detecções por 5min
    timeline_data = {}
    if ts_col and ts_col in detections.columns:
        try:
            df = detections.copy()
            df["_ts"] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
            df = df.dropna(subset=["_ts"])
            if len(df) > 0:
                df["_bucket"] = df["_ts"].dt.floor("5min")
                tl = df.groupby("_bucket").size()
                timeline_data = {
                    "x": [t.strftime("%H:%M") for t in tl.index],
                    "y": [int(v) for v in tl.values],
                }
        except Exception as e:
            print(f"Timeline error: {e}", file=sys.stderr)

    # distribuição de edge (histograma)
    edge_hist = {}
    if edge_col and edge_col in detections.columns:
        edges = detections[edge_col].dropna()
        edges_pct = edges * 100  # converte pra %
        edges_pct = edges_pct[(edges_pct >= -2) & (edges_pct <= 10)]  # remove outliers
        if len(edges_pct) > 0:
            hist, bin_edges = np.histogram(edges_pct, bins=40)
            edge_hist = {
                "bins": [f"{bin_edges[i]:.2f}" for i in range(len(hist))],
                "counts": [int(c) for c in hist],
            }

    # ===== HTML =====
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HFT Simulation Report — {run_dir.name}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0a0e1a;
    color: #e4e8f0;
    margin: 0;
    padding: 16px;
    max-width: 1000px;
    margin: 0 auto;
  }}
  h1 {{ color: #4ade80; border-bottom: 2px solid #4ade80; padding-bottom: 8px; }}
  h2 {{ color: #60a5fa; margin-top: 32px; border-bottom: 1px solid #334155; padding-bottom: 6px; }}
  h3 {{ color: #fbbf24; }}
  .meta {{ color: #94a3b8; font-size: 0.9em; margin-bottom: 24px; }}
  .card {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }}
  .kpi {{
    background: #0f172a;
    padding: 12px;
    border-radius: 6px;
    border-left: 4px solid #4ade80;
  }}
  .kpi .label {{ color: #94a3b8; font-size: 0.85em; text-transform: uppercase; }}
  .kpi .value {{ font-size: 1.8em; font-weight: bold; color: #e4e8f0; margin-top: 4px; }}
  .kpi.warn {{ border-left-color: #fbbf24; }}
  .kpi.bad {{ border-left-color: #ef4444; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
  }}
  th, td {{
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid #334155;
  }}
  th {{ background: #0f172a; color: #60a5fa; }}
  tr:hover td {{ background: #1e293b; }}
  .chart {{ min-height: 320px; margin: 16px 0; }}
  .verdict {{
    background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%);
    border-left: 4px solid #fbbf24;
    padding: 16px;
    border-radius: 6px;
    margin: 16px 0;
  }}
  .good {{ color: #4ade80; }}
  .warn {{ color: #fbbf24; }}
  .bad {{ color: #ef4444; }}
  code {{ background: #0f172a; padding: 2px 6px; border-radius: 3px; font-family: "SF Mono", Menlo, monospace; }}
  .sticky-nav {{
    position: sticky; top: 0; background: #0a0e1a; padding: 8px 0; margin-bottom: 16px;
    border-bottom: 1px solid #334155; z-index: 100;
  }}
  .sticky-nav a {{ color: #60a5fa; margin-right: 16px; text-decoration: none; font-size: 0.9em; }}
  .sticky-nav a:hover {{ color: #4ade80; }}
</style>
</head>
<body>

<h1>🔬 HFT Simulation Report</h1>
<div class="meta">
  Run: <code>{run_dir.name}</code><br>
  Gerado em: {gen_time}<br>
  Total detecções: <strong>{total_detections:,}</strong> | Total latency samples: <strong>{total_latency:,}</strong>
</div>

<div class="sticky-nav">
  <a href="#overview">Visão Geral</a>
  <a href="#latency">Latência</a>
  <a href="#edge">Edge Distribution</a>
  <a href="#fees">Após Fees</a>
  <a href="#timeline">Timeline</a>
  <a href="#verdict">Veredicto</a>
</div>

<h2 id="overview">📊 Visão Geral</h2>
<div class="card">
  <div class="grid">
    <div class="kpi">
      <div class="label">Total Detecções</div>
      <div class="value">{total_detections:,}</div>
    </div>
    <div class="kpi">
      <div class="label">Detecções/hora</div>
      <div class="value">{total_detections // 2 if total_detections > 0 else 0:,}</div>
    </div>
    <div class="kpi {'good' if edge_stats.get('median', 0) > 0.01 else 'warn'}">
      <div class="label">Edge mediano</div>
      <div class="value">{edge_stats.get('median', 0)*100:.2f}%</div>
    </div>
    <div class="kpi">
      <div class="label">Edge P90</div>
      <div class="value">{edge_stats.get('p90', 0)*100:.2f}%</div>
    </div>
    <div class="kpi">
      <div class="label">Edge máximo</div>
      <div class="value">{edge_stats.get('max', 0)*100:.2f}%</div>
    </div>
  </div>
</div>

<h2 id="latency">⏱️ Sobrevivência por Latência</h2>
<p>Das detecções, quantas ainda tinham edge após X ms de delay. Isso simula o tempo que você levaria para detectar + enviar a ordem.</p>
"""

    # tabela latência
    html += '<table><tr><th>Latência</th><th>Sobrevivem</th><th>Total</th><th>Taxa</th></tr>'
    for lat_label, lat_data in lat_stats.items():
        pct = lat_data['pct']
        color_class = 'good' if pct > 50 else ('warn' if pct > 25 else 'bad')
        html += f'<tr><td><strong>{lat_label}</strong></td><td>{lat_data["survived"]:,}</td><td>{lat_data["total"]:,}</td><td class="{color_class}"><strong>{pct}%</strong></td></tr>'
    html += '</table>'

    # chart latência
    if lat_stats:
        lat_labels_list = list(lat_stats.keys())
        lat_pcts = [lat_stats[k]["pct"] for k in lat_labels_list]
        html += f"""
<div id="chart-lat" class="chart"></div>
<script>
Plotly.newPlot('chart-lat', [{{
  x: {json.dumps(lat_labels_list)},
  y: {json.dumps(lat_pcts)},
  type: 'bar',
  marker: {{ color: ['#4ade80', '#60a5fa', '#fbbf24', '#f87171'] }},
  text: {json.dumps([f"{p}%" for p in lat_pcts])},
  textposition: 'auto',
}}], {{
  title: 'Taxa de sobrevivência por latência',
  paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a',
  font: {{ color: '#e4e8f0' }},
  xaxis: {{ title: 'Latência simulada' }},
  yaxis: {{ title: 'Taxa sobrevivência (%)', range: [0, 100] }}
}}, {{responsive: true}});
</script>
"""

    # edge distribution
    html += '<h2 id="edge">📈 Distribuição de Edge</h2>'
    html += '<p>Distribuição bruta do edge detectado (antes de fees). Valores em porcentagem.</p>'

    if bands:
        html += '<h3>Detecções por faixa de edge</h3>'
        html += '<table><tr><th>Faixa</th><th>Detecções</th><th>% do total</th></tr>'
        for band, count in bands.items():
            pct_band = 100.0 * count / total_detections if total_detections > 0 else 0
            html += f'<tr><td>{band}</td><td>{count:,}</td><td>{pct_band:.2f}%</td></tr>'
        html += '</table>'

        band_labels = list(bands.keys())
        band_counts = list(bands.values())
        html += f"""
<div id="chart-bands" class="chart"></div>
<script>
Plotly.newPlot('chart-bands', [{{
  x: {json.dumps(band_labels)},
  y: {json.dumps(band_counts)},
  type: 'bar',
  marker: {{ color: '#60a5fa' }},
}}], {{
  title: 'Detecções por faixa de edge bruto',
  paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a',
  font: {{ color: '#e4e8f0' }},
  xaxis: {{ title: 'Faixa de edge' }},
  yaxis: {{ title: 'Número de detecções' }}
}}, {{responsive: true}});
</script>
"""

    if edge_hist:
        html += f"""
<div id="chart-hist" class="chart"></div>
<script>
Plotly.newPlot('chart-hist', [{{
  x: {json.dumps(edge_hist["bins"])},
  y: {json.dumps(edge_hist["counts"])},
  type: 'bar',
  marker: {{ color: '#4ade80' }},
}}], {{
  title: 'Histograma detalhado de edge (em %)',
  paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a',
  font: {{ color: '#e4e8f0' }},
  xaxis: {{ title: 'Edge (%)' }},
  yaxis: {{ title: 'Frequência' }}
}}, {{responsive: true}});
</script>
"""

    # fees
    html += '<h2 id="fees">💰 Após Fees</h2>'
    html += '<p>Quantas detecções sobrevivem DEPOIS de descontar diferentes níveis de fee. <strong>Este é o número que importa.</strong></p>'

    if fee_scenarios:
        html += '<table><tr><th>Cenário</th><th>Net >0%</th><th>Net >0.5%</th><th>Net >1%</th><th>Net >2%</th></tr>'
        for scenario, data in fee_scenarios.items():
            html += f'<tr><td><strong>{scenario}</strong></td>'
            html += f'<td>{data.get("net_positive", 0):,}</td>'
            html += f'<td>{data.get("net_gt_0.5%", 0):,}</td>'
            html += f'<td>{data.get("net_gt_1%", 0):,}</td>'
            html += f'<td>{data.get("net_gt_2%", 0):,}</td>'
            html += '</tr>'
        html += '</table>'

        # chart
        scenarios = list(fee_scenarios.keys())
        net_pos = [fee_scenarios[s].get("net_positive", 0) for s in scenarios]
        net_05 = [fee_scenarios[s].get("net_gt_0.5%", 0) for s in scenarios]
        net_1 = [fee_scenarios[s].get("net_gt_1%", 0) for s in scenarios]
        net_2 = [fee_scenarios[s].get("net_gt_2%", 0) for s in scenarios]

        html += f"""
<div id="chart-fees" class="chart"></div>
<script>
Plotly.newPlot('chart-fees', [
  {{ x: {json.dumps(scenarios)}, y: {json.dumps(net_pos)}, name: 'Net >0%', type: 'bar', marker: {{ color: '#4ade80' }} }},
  {{ x: {json.dumps(scenarios)}, y: {json.dumps(net_05)}, name: 'Net >0.5%', type: 'bar', marker: {{ color: '#60a5fa' }} }},
  {{ x: {json.dumps(scenarios)}, y: {json.dumps(net_1)}, name: 'Net >1%', type: 'bar', marker: {{ color: '#fbbf24' }} }},
  {{ x: {json.dumps(scenarios)}, y: {json.dumps(net_2)}, name: 'Net >2%', type: 'bar', marker: {{ color: '#f87171' }} }}
], {{
  title: 'Oportunidades líquidas por cenário de fee',
  paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a',
  font: {{ color: '#e4e8f0' }},
  barmode: 'group',
  xaxis: {{ title: 'Cenário' }},
  yaxis: {{ title: 'Detecções' }}
}}, {{responsive: true}});
</script>
"""

    # timeline
    if timeline_data:
        html += '<h2 id="timeline">📅 Timeline</h2>'
        html += '<p>Detecções ao longo da simulação (buckets de 5min).</p>'
        html += f"""
<div id="chart-timeline" class="chart"></div>
<script>
Plotly.newPlot('chart-timeline', [{{
  x: {json.dumps(timeline_data["x"])},
  y: {json.dumps(timeline_data["y"])},
  type: 'scatter',
  mode: 'lines+markers',
  line: {{ color: '#4ade80', width: 2 }},
  marker: {{ size: 6 }},
  fill: 'tozeroy',
  fillcolor: 'rgba(74, 222, 128, 0.1)',
}}], {{
  title: 'Detecções ao longo do tempo',
  paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a',
  font: {{ color: '#e4e8f0' }},
  xaxis: {{ title: 'Hora' }},
  yaxis: {{ title: 'Detecções por 5min' }}
}}, {{responsive: true}});
</script>
"""

    # categorias
    if cat_counts:
        html += '<h2>🏷️ Por Categoria</h2>'
        cats = list(cat_counts.keys())
        counts = list(cat_counts.values())
        html += f"""
<div id="chart-cats" class="chart"></div>
<script>
Plotly.newPlot('chart-cats', [{{
  labels: {json.dumps(cats)},
  values: {json.dumps(counts)},
  type: 'pie',
  hole: 0.4,
}}], {{
  title: 'Distribuição por categoria',
  paper_bgcolor: '#0a0e1a', plot_bgcolor: '#0a0e1a',
  font: {{ color: '#e4e8f0' }}
}}, {{responsive: true}});
</script>
"""

    # veredicto
    html += '<h2 id="verdict">🎯 Veredicto Honesto</h2>'

    # calcula veredicto baseado em fees
    verdict_text = []
    if fee_scenarios:
        fee_2pct = fee_scenarios.get("Fee 2% (Poly típico)", {})
        fee_3pct = fee_scenarios.get("Fee 3%", {})
        fee_7pct = fee_scenarios.get("Fee 7% (Crypto c/ taker fee)", {})

        net_after_2_gt_0 = fee_2pct.get("net_positive", 0)
        net_after_2_gt_1 = fee_2pct.get("net_gt_1%", 0)
        net_after_7_gt_0 = fee_7pct.get("net_positive", 0)

        per_hour_2 = net_after_2_gt_0 // 2
        per_hour_2_gt1 = net_after_2_gt_1 // 2

        verdict_text.append(f"<p><strong>Com fee 2% (mercados Polymarket típicos):</strong></p>")
        verdict_text.append(f"<ul>")
        verdict_text.append(f"<li>{net_after_2_gt_0:,} detecções com edge líquido > 0 ({per_hour_2}/hora)</li>")
        verdict_text.append(f"<li>{net_after_2_gt_1:,} com edge líquido > 1% ({per_hour_2_gt1}/hora)</li>")
        verdict_text.append(f"</ul>")

        verdict_text.append(f"<p><strong>Com fee 7% (crypto markets com taker fee):</strong></p>")
        verdict_text.append(f"<ul>")
        verdict_text.append(f"<li>{net_after_7_gt_0:,} detecções com edge líquido > 0 ({net_after_7_gt_0 // 2}/hora)</li>")
        verdict_text.append(f"</ul>")

        if per_hour_2_gt1 > 20:
            verdict_text.append('<div class="verdict"><p class="good"><strong>✅ Sinal positivo.</strong> Mesmo descontando fees típicas de 2%, há volume significativo de oportunidades com edge líquido relevante (>1%).</p></div>')
        elif per_hour_2_gt1 > 5:
            verdict_text.append('<div class="verdict"><p class="warn"><strong>⚠️ Sinal misto.</strong> Existem oportunidades mas em volume moderado. Precisa considerar se a infraestrutura compensa.</p></div>')
        else:
            verdict_text.append('<div class="verdict"><p class="bad"><strong>❌ Sinal fraco.</strong> Após fees típicas, poucas oportunidades sobrevivem. Estratégia questionável.</p></div>')

    html += "\n".join(verdict_text)

    html += """
<h3>⚠️ Importante — O que esta análise NÃO testa</h3>
<div class="card">
<ul>
  <li><strong>Slippage real na execução:</strong> o book mostrado é passivo. Executar consome liquidez e o preço se move contra você.</li>
  <li><strong>Competição:</strong> bots com latência sub-50ms vão capturar as melhores oportunidades antes de você.</li>
  <li><strong>Fill probability:</strong> não é porque o edge existe que a ordem é filled. Muitas serão canceladas antes.</li>
  <li><strong>Gas/settlement real:</strong> em NegRisk, o settle tem custos adicionais de gas e tempo.</li>
  <li><strong>Modelagem de quote update:</strong> o edge detectado pode ser estale (dados antigos ainda no book).</li>
</ul>
<p>Pra ter certeza, o próximo passo é rodar em <strong>dry-run mode executando papel</strong> (sem dinheiro real), comparando detecções vs. trades que realmente seriam fillados.</p>
</div>

</body>
</html>
"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Nome do run (ex: run_2026-04-20_00h19). Default: último.")
    ap.add_argument("--output-dir", default=str(Path.home() / "hft" / "output"))
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if args.run:
        run_dir = output_dir / args.run
    else:
        run_dir = find_latest_run(output_dir)

    if not run_dir.exists():
        raise SystemExit(f"Run não encontrado: {run_dir}")

    print(f"Analisando: {run_dir}")

    detections = load_all_parquets(run_dir, "detections")
    latency = load_all_parquets(run_dir, "latency")

    print(f"Detecções: {len(detections):,}")
    print(f"Latency samples: {len(latency):,}")

    if len(detections) > 0:
        print(f"Colunas detections: {list(detections.columns)}")
    if len(latency) > 0:
        print(f"Colunas latency: {list(latency.columns)}")

    html = build_html(run_dir, detections, latency)
    out_path = run_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Relatório gerado: {out_path}")
    print(f"Tamanho: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"\nPara baixar pro celular e abrir no navegador:")
    print(f"  scp root@SEU_IP:{out_path} ~/Downloads/")
    print(f"Ou pegue o conteúdo via:")
    print(f"  cat {out_path}")


if __name__ == "__main__":
    main()
