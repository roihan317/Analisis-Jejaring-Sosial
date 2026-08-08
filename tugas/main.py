"""Analisis Jejaring Sosial: Email-Eu-core (1.005 node).

Menjalankan seluruh jawaban proyek: pemuatan data, adjacency matrix,
centrality, metrik global, Louvain, simulasi SIR, visualisasi, dan GEXF.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_URL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
EXPECTED_NODES = 1005


def download_dataset(data_path: Path) -> None:
    """Download SNAP Email-Eu-core bila berkas belum ada."""
    if data_path.exists():
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh dataset dari {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, data_path)


def load_graph(data_path: Path) -> nx.DiGraph:
    """Muat dataset sebagai graf berarah dan tidak berbobot.

    Setiap baris u v berarti pengirim u mengirim email kepada penerima v.
    Kehadiran edge bernilai 1; frekuensi tidak tersedia pada dataset asli.
    """
    graph = nx.DiGraph()
    with gzip.open(data_path, "rt", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            if u != v:
                graph.add_edge(u, v, weight=1)
    # Dataset SNAP menyatakan 1.005 personel; tambahkan node tanpa edge bila ada.
    graph.add_nodes_from(range(EXPECTED_NODES))
    return graph


def top_items(values: dict, n: int = 10) -> list[tuple[int, float]]:
    return [(int(node), float(value)) for node, value in
            sorted(values.items(), key=lambda item: (-item[1], item[0]))[:n]]


def compute_centralities(graph: nx.DiGraph) -> dict[str, dict]:
    """Hitung empat metrik centrality pada graf berarah."""
    print("Menghitung centrality (betweenness exact dapat memerlukan beberapa menit)...")
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    closeness = nx.closeness_centrality(graph)  # kedekatan node terhadap pengirim lain
    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=2000, tol=1e-8)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(graph)
    return {
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
    }


def global_metrics(graph: nx.DiGraph) -> tuple[nx.Graph, dict]:
    """Metrik global dihitung pada komponen terhubung terbesar versi tak-berarah.

    Konversi ini lazim dipakai agar diameter dan average path length terdefinisi,
    sekaligus mengukur kohesi struktural tanpa mempersoalkan arah pesan.
    """
    undirected = nx.Graph(graph)
    giant_nodes = max(nx.connected_components(undirected), key=len)
    giant = undirected.subgraph(giant_nodes).copy()
    return giant, {
        "nodes_total": graph.number_of_nodes(),
        "edges_directed": graph.number_of_edges(),
        "weak_components": nx.number_weakly_connected_components(graph),
        "giant_component_nodes": giant.number_of_nodes(),
        "giant_component_edges": giant.number_of_edges(),
        "density_directed": nx.density(graph),
        "density_undirected_giant": nx.density(giant),
        "diameter_giant": nx.diameter(giant),
        "average_path_length_giant": nx.average_shortest_path_length(giant),
        "average_clustering_giant": nx.average_clustering(giant),
        "transitivity_giant": nx.transitivity(giant),
    }


def detect_louvain(giant: nx.Graph, seed: int) -> tuple[list[set], dict]:
    communities = nx.community.louvain_communities(giant, weight="weight", seed=seed)
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    membership = {node: i for i, community in enumerate(communities) for node in community}
    modularity = nx.community.modularity(giant, communities, weight="weight")
    summary = {
        "algorithm": "Louvain (NetworkX)",
        "community_count": len(communities),
        "modularity": modularity,
        "largest_community_size": len(communities[0]),
        "community_sizes": [len(c) for c in communities],
    }
    return communities, {"membership": membership, "summary": summary}


def sir_simulation(graph: nx.DiGraph, seed_node: int, beta: float, gamma: float,
                   steps: int, rng: random.Random) -> pd.DataFrame:
    """Model SIR diskret; transmisi mengikuti arah pengirim -> penerima."""
    susceptible = set(graph.nodes()) - {seed_node}
    infected = {seed_node}
    recovered: set[int] = set()
    rows = []
    for step in range(steps + 1):
        rows.append({"step": step, "S": len(susceptible), "I": len(infected),
                     "R": len(recovered), "ever_reached": len(infected | recovered)})
        new_infected = set()
        for node in infected:
            for neighbor in graph.successors(node):
                if neighbor in susceptible and rng.random() < beta:
                    new_infected.add(neighbor)
        new_recovered = {node for node in infected if rng.random() < gamma}
        susceptible -= new_infected
        infected = (infected | new_infected) - new_recovered
        recovered |= new_recovered
        if not infected and not new_infected:
            break
    return pd.DataFrame(rows)


def run_sir_comparison(graph: nx.DiGraph, top_node: int, repetitions: int,
                       beta: float, gamma: float, steps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bandingkan seed degree-tertinggi dengan seed acak secara berulang."""
    nodes = sorted(graph.nodes())
    rows, trajectories = [], []
    master_rng = random.Random(seed)
    for strategy in ("top_degree", "random"):
        for run in range(repetitions):
            initial = top_node if strategy == "top_degree" else master_rng.choice(nodes)
            result = sir_simulation(graph, initial, beta, gamma, steps,
                                    random.Random(master_rng.randrange(2**32)))
            result["strategy"], result["run"], result["seed_node"] = strategy, run, initial
            trajectories.append(result)
            peak_row = result.loc[result["I"].idxmax()]
            reached = int(result["ever_reached"].max())
            rows.append({"strategy": strategy, "run": run, "seed_node": initial,
                         "final_reached": reached, "reach_fraction": reached / graph.number_of_nodes(),
                         "peak_infected": int(peak_row["I"]), "peak_step": int(peak_row["step"]),
                         "duration_steps": int(result["step"].max())})
    return pd.DataFrame(rows), pd.concat(trajectories, ignore_index=True)


def save_visualizations(graph: nx.DiGraph, giant: nx.Graph, centrality: dict,
                        membership: dict, sir_trajectory: pd.DataFrame, out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Sampel visual agar 1.005 node tetap terbaca.
    sample_nodes = sorted(giant, key=lambda n: (-centrality["degree"].get(n, 0), n))[:250]
    sample = giant.subgraph(sample_nodes)
    layout = nx.spring_layout(sample, seed=seed, k=0.28)
    sizes = [70 + 1800 * centrality["degree"].get(n, 0) for n in sample]
    colors = [membership.get(n, -1) for n in sample]
    plt.figure(figsize=(12, 9))
    nx.draw_networkx_edges(sample, layout, alpha=0.14, width=0.45, edge_color="#64748b")
    nodes = nx.draw_networkx_nodes(sample, layout, node_size=sizes, node_color=colors,
                                   cmap="tab20", alpha=0.88, linewidths=0.2, edgecolors="white")
    plt.colorbar(nodes, label="ID komunitas Louvain")
    plt.title("Subgraf 250 Node Paling Terhubung - Email-Eu-core")
    plt.axis("off"); plt.tight_layout(); plt.savefig(out / "network_communities.png", dpi=220); plt.close()

    mean_curve = sir_trajectory.groupby(["strategy", "step"], as_index=False)["ever_reached"].mean()
    plt.figure(figsize=(10, 5.5))
    for strategy, group in mean_curve.groupby("strategy"):
        label = "Seed top degree centrality" if strategy == "top_degree" else "Seed acak"
        plt.plot(group["step"], group["ever_reached"], marker="o", markersize=3, label=label)
    plt.title("Rata-rata Jangkauan Kumulatif Simulasi SIR")
    plt.xlabel("Langkah waktu"); plt.ylabel("Node yang pernah terjangkau")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "sir_comparison.png", dpi=220); plt.close()

    # GEXF ekspor ke Gephi; centrality dan komunitas ditambahkan sebagai atribut node.
    export = graph.copy()
    for node in export:
        export.nodes[node].update({metric: float(values[node]) for metric, values in centrality.items()})
        export.nodes[node]["community"] = int(membership.get(node, -1))
    nx.write_gexf(export, out / "email_eu_core_analysis.gexf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Proyek Social Network Analysis Email-Eu-core")
    parser.add_argument("--output", default="results", help="folder hasil analisis")
    parser.add_argument("--data", default="data/email-Eu-core.txt.gz", help="berkas dataset .gz")
    parser.add_argument("--sir-runs", type=int, default=30)
    parser.add_argument("--sir-steps", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.12)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out, data_path = Path(args.output), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    download_dataset(data_path)
    graph = load_graph(data_path)
    centrality = compute_centralities(graph)
    giant, metrics = global_metrics(graph)
    communities, community_data = detect_louvain(giant, args.seed)
    membership, community_summary = community_data["membership"], community_data["summary"]

    adjacency_nodes = sorted(graph.nodes())[:5]
    adjacency = nx.to_pandas_adjacency(graph, nodelist=adjacency_nodes, dtype=int, weight=None)
    adjacency.to_csv(out / "adjacency_matrix_5_nodes.csv")
    centrality_table = pd.DataFrame({name: pd.Series(values) for name, values in centrality.items()})
    centrality_table.index.name = "node"
    centrality_table.sort_values("degree", ascending=False).to_csv(out / "centrality_all_nodes.csv")
    top = {name: top_items(values) for name, values in centrality.items()}
    top_degree_node = top["degree"][0][0]
    sir_summary, sir_trajectory = run_sir_comparison(graph, top_degree_node, args.sir_runs,
                                                      args.beta, args.gamma, args.sir_steps, args.seed)
    sir_summary.to_csv(out / "sir_runs.csv", index=False)
    sir_trajectory.to_csv(out / "sir_trajectories.csv", index=False)
    sir_aggregate = sir_summary.groupby("strategy").agg(
        mean_final_reached=("final_reached", "mean"), mean_reach_fraction=("reach_fraction", "mean"),
        mean_peak_infected=("peak_infected", "mean"), mean_peak_step=("peak_step", "mean"),
        mean_duration=("duration_steps", "mean"), runs=("run", "count")).reset_index()
    sir_aggregate.to_csv(out / "sir_summary.csv", index=False)
    save_visualizations(graph, giant, centrality, membership, sir_trajectory, out, args.seed)
    report = {"dataset": {"name": "Email-Eu-core", "source": DATA_URL,
                          "representation": "directed, unweighted"}, "global_metrics": metrics,
              "top_centralities": top, "communities": community_summary,
              "adjacency_nodes": adjacency_nodes, "sir_parameters": {"beta": args.beta, "gamma": args.gamma,
                  "runs_per_strategy": args.sir_runs, "max_steps": args.sir_steps, "seed": args.seed},
              "sir_comparison": sir_aggregate.to_dict(orient="records")}
    with open(out / "analysis_summary.json", "w", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Selesai. Hasil tersimpan di: {out.resolve()}")


if __name__ == "__main__":
    main()
