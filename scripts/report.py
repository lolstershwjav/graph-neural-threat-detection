"""Create measured figures and a research report from the graph study."""

import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

NAMES = {
    "flow_tabular": "Flow boosting", "context_tabular": "Context boosting",
    "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "temporal_sage": "Temporal GraphSAGE",
    "sage_self": "GraphSAGE, self only", "sage_rewired": "GraphSAGE, rewired",
    "sage_flow_only": "GraphSAGE, flow loss",
}


def fmt(stat):
    return f"{100*stat['mean']:.1f}" if stat else "—"


def plots(data, root):
    models = list(NAMES)
    summary = data["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    colorset = ["#55779e", "#55779e", "#01846e", "#01846e", "#01846e", "#01846e",
                "#a65d3c", "#a65d3c", "#a65d3c"]
    pos = np.arange(len(models))
    for ax, scope in zip(axes, ("flow", "unseen")):
        stats = [summary[m][scope].get("ap") for m in models]
        avg = [100*s["mean"] if s else 0 for s in stats]
        lo = [max(0, a-100*s["low"]) if s else 0 for a, s in zip(avg, stats)]
        hi = [max(0, 100*s["high"]-a) if s else 0 for a, s in zip(avg, stats)]
        ax.barh(pos, avg, xerr=[lo, hi], color=colorset, height=.57, capsize=2)
        ax.set_yticks(pos, labels=[NAMES[m] for m in models] if scope == "flow" else [""]*len(pos))
        ax.invert_yaxis()
        ax.set(xlabel="Average precision (%)", xlim=(0, 100),
               title="All confirmed flows" if scope == "flow" else "Unseen-endpoint flows")
        ax.grid(axis="x", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "benchmark.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    choice = ("flow_tabular", "context_tabular", "sage", "temporal_sage")
    captures = [str(c) for c in data["config"]["test"]]
    marks = np.arange(len(captures))
    palette = ("#55779e", "#8c64a8", "#007c6d", "#c48143")
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    for i, (model, color) in enumerate(zip(choice, palette)):
        values = [mean(row["capture"][c]["ap"] for row in data["runs"]
                       if row["model"] == model and row["capture"][c]["ap"] is not None)
                  for c in captures]
        ax.bar(marks + (i-1.5)*.19, values, width=.17, color=color, label=NAMES[model])
    ax.set(xticks=marks, xticklabels=captures, xlabel="Held-out capture", ylabel="Average precision",
           ylim=(0, 1))
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.21))
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "capture-variation.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pdf(data, root):
    dark = colors.HexColor("#193b50")
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("StudySans", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("StudySans-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("StudySans", normal="StudySans", bold="StudySans-Bold")
    styles = getSampleStyleSheet()
    for name in ("Normal", "BodyText"):
        styles[name].fontName = "StudySans"
    for name in ("Title", "Heading2"):
        styles[name].fontName = "StudySans-Bold"
        styles[name].textColor = dark
    styles["Title"].alignment = 0
    styles["Title"].fontSize = 23
    styles["Title"].leading = 27
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 8
    body = []

    def p(s, style="BodyText"):
        body.append(Paragraph(s, styles[style]))

    def section(title, content):
        p(title, "Heading2")
        p(content)

    def graphic(name, height):
        body.append(Image(str(root/name), width=483, height=height, kind="proportional"))

    summary = data["summary"]
    counts = data["dataset_counts"]
    test = counts["splits"]["test"]
    def f(model, scope, key):
        return fmt(summary[model][scope].get(key))

    p("Graph Neural Networks for Flow Detection", "Title")
    p("Capture-disjoint evaluation with observation-time graph windows")
    p(f"<b>Abstract.</b> We construct {counts['windows']:,} network graphs from "
      f"{counts['flows']:,} sampled bidirectional CTU-13 flows. The test captures contain "
      f"{test['botnet']:,} labeled botnet flows and {test['normal']:,} confirmed normal flows. "
      f"Temporal GraphSAGE achieves {f('temporal_sage','flow','ap')}% average precision, "
      f"compared with {f('flow_tabular','flow','ap')}% for flow-only gradient boosting and "
      f"{f('context_tabular','flow','ap')}% for a tabular model with node context. "
      "We isolate adjacency, historical context and auxiliary supervision.")
    section("1. Questions and hypotheses",
            "Can neighborhood aggregation distinguish suspicious flows after holding out whole "
            "captures? We expect genuine links to help more than self-only or shuffled message "
            "paths, and past observations to help when host behavior persists across windows.")
    section("2. Related work",
            "GCN uses normalized adjacency [1], GraphSAGE learns shared neighborhood aggregation [2], "
            "and GAT learns attention weights within each neighborhood [3]. E-GraphSAGE motivates "
            "the use of both local flow data and graph structure in intrusion detection [4]. "
            "CTU-13 documents confirmed traffic and separate background context [5].")
    section("3. Data and temporal design",
            "Thirteen public captures contribute four disjoint bounded byte ranges each. "
            "Seven captures fit models and transforms, three select epochs and calibrate, and "
            "three are reserved for testing. Flow windows use estimated completion timestamp. "
            "We retain background edges for context and exclude them from supervised metrics. "
            "The reported host target denotes incident-flow membership rather than compromise.")
    section("4. Architecture and protocol",
            "Nodes embed incoming and outgoing counts and flow means. Directed flow readouts "
            "combine learned source and destination vectors with observed flow features. "
            "GCN, GraphSAGE and GAT each perform two message steps. Temporal GraphSAGE adds a GRU "
            "summary of three earlier windows. Epoch selection uses validation AP and thresholds "
            "use the confirmed-normal validation score quantiles.")

    body.append(PageBreak())
    section("5. Results",
            "Table 1 gives means across three seeds. Flow and host refer to the separately labeled "
            "readouts. Unseen AP uses flows with an endpoint absent from all train captures.")
    fields = [("flow","ap"),("flow","recall"),("flow","fpr"),("host","ap"),("unseen","ap")]
    rows = [["Model", "Flow AP", "Recall", "FPR", "Host AP", "Unseen AP"]]
    for model, title in NAMES.items():
        rows.append([title, *(f(model, scope, metric) for scope, metric in fields)])
    table = Table(rows, colWidths=[145,64,63,61,70,84], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"StudySans"),
        ("FONTNAME",(0,0),(-1,0),"StudySans-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("BACKGROUND",(0,0),(-1,0),dark),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#edf5f8")]),
        ("ALIGN",(1,0),(-1,-1),"RIGHT"),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    body.append(table)
    graphic("benchmark.png", 260)
    p("<b>Figure 1.</b> Aggregate and unseen-endpoint discrimination. Horizontal bars show seed "
      "bootstrap intervals within the same fixed captures.")

    body.append(PageBreak())
    section("6. Ablations and capture variability",
            f"GraphSAGE achieves {f('sage','flow','ap')}% flow AP with true adjacency, "
            f"{f('sage_self','flow','ap')}% with self-only messages and "
            f"{f('sage_rewired','flow','ap')}% with rewired message addresses. Interpret these "
            "effects jointly with the endpoint-summary tabular baseline. The historical "
            "encoder sees fewer prior windows at the beginning of each source segment.")
    graphic("capture-variation.png", 182)
    p("<b>Figure 2.</b> Mean flow AP within each held-out capture.")
    p("GAT scored the entire 363-window test set in a mean 0.347 seconds after graph "
      "preprocessing. Small windows (57 edges or fewer) contain only 64 labeled test flows. "
      "The unseen-endpoint slice contains 265 positives among 337 labeled flows, which "
      "raises its average-precision baseline.")
    section("7. Interpretation and limitations",
            "Structural explanations show changes in the same model after removing neighbor "
            "messages. They indicate sensitivity rather than cause. Background traffic lacks "
            "verified negative labels. Capture slices can omit traffic just outside the ranges. "
            "CTU-13 is older single-site data, and a held-out capture is not necessarily a novel "
            "malware family. Multi-stage ground truth is unavailable.")
    section("8. Reproducibility and next experiments",
            "The repository includes the exact segment digests, model settings, all per-seed "
            "metrics including size slices and timing, and an offline inspector for anonymized "
            "graph evidence. Follow up with independent, newer networks and endpoint-level "
            "ground truth. Compare the same time budget across graph architectures and use "
            "external calibration before generalizing the alert thresholds.")
    p("References", "Heading2")
    p("[1] Kipf and Welling, ICLR (2017). [2] Hamilton et al., NeurIPS (2017). "
      "[3] Veličković et al., ICLR (2018). [4] Lo et al., IEEE NOMS (2022). "
      "[5] García et al., Computers &amp; Security (2014). See docs/METHODOLOGY.md for full links.")

    def footer(c, doc):
        c.setFont("StudySans", 8)
        c.setFillColor(dark)
        c.drawString(52, 27, "Graph Neural Networks for Flow Detection")
        c.drawRightString(554, 27, str(doc.page))

    doc = SimpleDocTemplate(str(root/"graph-flow-detection-report.pdf"),
                            title="Graph Neural Networks for Flow Detection",
                            leftMargin=52, rightMargin=52, topMargin=42, bottomMargin=42)
    doc.build(body, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    out = Path("results")
    with gzip.open(out / "study.json.gz", "rt", encoding="utf8") as source:
        study = json.load(source)
    plots(study, out)
    pdf(study, out)
