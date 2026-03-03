import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages


# =========================
# Configurações
# =========================
LABELS = ["COVID-19", "Influenza A", "Influenza B", "Saudável"]

# Caminhos robustos: relativos ao arquivo se.py (não ao "cwd" do terminal)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]  # projeto-diagnostico-se/
DATA_PATH = PROJECT_ROOT / "data" / "dataset_filtrado.csv"

OUT_PDF = PROJECT_ROOT / "src" / "scripts" / "se" / "relatorio_validacao_se.pdf"


# =========================
# Helpers de plot
# =========================

def plot_cm(cm_data, labels, title, fmt_int=True, vmin=None, vmax=None, log_scale=False):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, 7))

    # Use pcolormesh instead of imshow — draws clean cell boundaries naturally
    if log_scale:
        # Mask zeros so LogNorm doesn't break; display them as the minimum color
        data_plot = np.where(cm_data == 0, np.nan, cm_data.astype(float))
        norm = mcolors.LogNorm(
            vmin=max(float(np.nanmin(data_plot[~np.isnan(data_plot)])), 1.0),
            vmax=float(np.nanmax(cm_data))
        )
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color="#e8f0f7")   # zeros get lightest blue
        im = ax.pcolormesh(data_plot, cmap=cmap, norm=norm, edgecolors="white", linewidth=2)
        cbar_label = "Contagem (escala log)"
    else:
        im = ax.pcolormesh(
            cm_data.astype(float),
            cmap="Blues",
            vmin=vmin if vmin is not None else 0,
            vmax=vmax if vmax is not None else 1,
            edgecolors="white",
            linewidth=2
        )
        cbar_label = "Proporção (recall)"

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)

    # pcolormesh origin is bottom-left, so we flip the y-axis to match
    # matrix convention (row 0 at top)
    ax.invert_yaxis()

    # Center ticks on each cell
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    ax.set_xlabel("Predito", fontsize=12, labelpad=8)
    ax.set_ylabel("Verdadeiro", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, pad=14)

    # Remove outer spines for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotations — centered in each cell
    thresh = float(np.nanmax(cm_data)) / 2.0 if np.nanmax(cm_data) else 0.0
    for i in range(n):
        for j in range(n):
            val = cm_data[i, j]
            txt = f"{int(val)}" if fmt_int else f"{val:.2%}"
            color = "white" if val > thresh else "black"
            ax.text(
                j + 0.5, i + 0.5, txt,
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=color
            )

    fig.tight_layout()
    return fig

def plot_report_table(report_dict, labels):
    rows = []
    for lab in labels:
        d = report_dict.get(lab, {})
        rows.append({
            "Classe": lab,
            "Precision": f"{d.get('precision', 0):.3f}",
            "Recall": f"{d.get('recall', 0):.3f}",
            "F1-score": f"{d.get('f1-score', 0):.3f}",
            "Support": str(int(d.get("support", 0))),
        })

    # separador visual
    rows.append({
        "Classe": "────────────",
        "Precision": "────────",
        "Recall": "────────",
        "F1-score": "────────",
        "Support": "────────",
    })

    for agg in ["accuracy", "macro avg", "weighted avg"]:
        d = report_dict.get(agg, {})
        if agg == "accuracy":
            rows.append({
                "Classe": "accuracy",
                "Precision": "",
                "Recall": "",
                "F1-score": f"{d:.3f}" if isinstance(d, (int, float)) else "",
                "Support": str(int(report_dict["macro avg"]["support"])),
            })
        else:
            rows.append({
                "Classe": agg,
                "Precision": f"{d.get('precision', 0):.3f}",
                "Recall": f"{d.get('recall', 0):.3f}",
                "F1-score": f"{d.get('f1-score', 0):.3f}",
                "Support": str(int(d.get("support", 0))),
            })

    df_rep = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    ax.set_title("Relatório de Classificação", fontsize=14, pad=12)

    tbl = ax.table(
        cellText=df_rep.values,
        colLabels=df_rep.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)

    # cabeçalho estilizado
    for j in range(len(df_rep.columns)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # separador em cinza
    sep_row = len(labels) + 1
    for j in range(len(df_rep.columns)):
        tbl[sep_row, j].set_facecolor("#dddddd")

    fig.tight_layout()
    return fig


def plot_legend():
    texto = (
        "Legenda — Relatório de Classificação\n\n"
        "precision (Precisão / PPV)\n"
        "  Entre os casos previstos como uma classe, qual fração realmente pertence a ela.\n\n"
        "recall (Sensibilidade / TPR)\n"
        "  Entre os casos realmente de uma classe, qual fração foi corretamente identificada.\n\n"
        "f1-score\n"
        "  Média harmônica entre precision e recall. Penaliza quando um dos dois é muito baixo.\n\n"
        "support\n"
        "  Número de amostras reais daquela classe no dataset.\n\n"
        "Linhas agregadas:\n"
        "  accuracy      : fração total de acertos.\n"
        "  macro avg     : média simples entre classes (cada classe pesa igual).\n"
        "  weighted avg  : média ponderada pelo support (classes frequentes pesam mais).\n\n"
        "Matriz de Confusão\n"
        "  Linhas = classe verdadeira | Colunas = classe predita.\n"
        "  Diagonal = acertos | Fora da diagonal = erros.\n"
        "  Versão normalizada: proporção por linha (equivalente ao recall por classe).\n"
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")
    ax.set_title("Explicação das Métricas", fontsize=14, pad=12)
    ax.text(
        0.02, 0.97, texto,
        va="top", ha="left",
        fontsize=10.5,
        family="monospace",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f0f4f8",
            edgecolor="#aac4de",
        ),
    )
    fig.tight_layout()
    return fig


def plot_summary(acc, y_true, y_pred):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    ax.set_title("Resumo da Validação — Sistema Especialista", fontsize=14, pad=12)

    texto = (
        f"  Dataset: {DATA_PATH}\n"
        f"  Total de amostras : {len(y_true)}\n"
        f"  Acurácia overall  : {acc:.4f}  ({acc*100:.2f}%)\n\n"
        "  Distribuição das classes — Verdadeiro:\n"
        + "\n".join(f"      {k:<15} {v:>5}" for k, v in y_true.value_counts().items())
        + "\n\n"
        "  Distribuição das classes — Predito:\n"
        + "\n".join(f"      {k:<15} {v:>5}" for k, v in y_pred.value_counts().items())
    )

    ax.text(
        0.02, 0.97, texto,
        va="top", ha="left",
        fontsize=11,
        family="monospace",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f0f4f8",
            edgecolor="#aac4de",
        ),
    )
    fig.tight_layout()
    return fig


# =========================
# Pipeline principal
# =========================
df = pd.read_csv(DATA_PATH, decimal=",")
df.columns = df.columns.str.replace("\xa0", " ").str.strip()

colunas = [
    "SARS-Cov-2 exam result",
    "Influenza A, rapid test",
    "Influenza B, rapid test",
    "Lactic Dehydrogenase",
    "Mean corpuscular hemoglobin concentration (MCHC)",
    "Proteina C reativa mg/dL",
    "Leukocytes",
    "Lymphocytes",
    "Platelets",
    "Monocytes",
    "Neutrophils",
    "Red blood Cells",
]

df = df[colunas].copy()
df.fillna(df.mean(numeric_only=True), inplace=True)

variaveis_laboratoriais = [
    "Lactic Dehydrogenase",
    "Mean corpuscular hemoglobin concentration (MCHC)",
    "Proteina C reativa mg/dL",
    "Leukocytes",
    "Lymphocytes",
    "Platelets",
    "Monocytes",
    "Neutrophils",
    "Red blood Cells",
]

scaler = StandardScaler()
df[variaveis_laboratoriais] = scaler.fit_transform(df[variaveis_laboratoriais])


def classe_real(row):
    if row["SARS-Cov-2 exam result"] == "positive":
        return "COVID-19"
    elif row["Influenza A, rapid test"] == "positive":
        return "Influenza A"
    elif row["Influenza B, rapid test"] == "positive":
        return "Influenza B"
    else:
        return "Saudável"


def sistema_especialista(row):
    score_covid = 0
    score_flu_a = 0
    score_flu_b = 0

    # ── COVID-19: 5 signals, threshold = 3 (stricter → fewer false positives) ──
    # Real COVID signals: Leukocytes mean=-0.721, Platelets mean=-0.706
    if row["Lactic Dehydrogenase"] > 0.5:      score_covid += 1
    if row["Proteina C reativa mg/dL"] > 0.5:  score_covid += 1
    if row["Neutrophils"] > 0.5:               score_covid += 1
    if row["Leukocytes"] < -0.5:               score_covid += 1  # strongest COVID signal
    if row["Platelets"] < -0.5:                score_covid += 1  # strongest COVID signal

    # ── Influenza A: threshold = 1 (any strong signal) ────────────────────────
    # Real Flu A: Lymphocytes mean=-1.061, Neutrophils mean=+1.140
    if row["Lymphocytes"] < -0.7:              score_flu_a += 1
    if row["Neutrophils"] > 0.7:               score_flu_a += 1

    # ── Influenza B: threshold = 1 (any strong signal) ────────────────────────
    # Real Flu B: Proteina C reativa mean=+0.710, Platelets mean=-0.396
    if row["Monocytes"] > 0.5:                 score_flu_b += 1
    if row["Red blood Cells"] < -0.5:          score_flu_b += 1
    if row["Proteina C reativa mg/dL"] > 0.5:  score_flu_b += 1  # strongest Flu B signal

    # COVID checked first — it has the most cases and strongest signals
    if score_covid >= 3:   return "COVID-19"
    elif score_flu_a >= 1: return "Influenza A"
    elif score_flu_b >= 1: return "Influenza B"
    else:                  return "Saudável"

def diagnosticar(dados_paciente):
    """
    Recebe um dicionário com exames laboratoriais,
    aplica o mesmo scaler e retorna o diagnóstico.
    """

    entrada = pd.DataFrame([dados_paciente])

    # Normaliza usando o MESMO scaler treinado
    entrada[variaveis_laboratoriais] = scaler.transform(
        entrada[variaveis_laboratoriais]
    )

    resultado = sistema_especialista(entrada.iloc[0])

    return resultado


df["Classe_real"] = df.apply(classe_real, axis=1)
df["Predicao_SE"] = df.apply(sistema_especialista, axis=1)

y_true = df["Classe_real"]
y_pred = df["Predicao_SE"]

acc = accuracy_score(y_true, y_pred)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(y_true, y_pred, labels=LABELS)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# =========================
# Gera PDF (5 páginas)
# =========================
with PdfPages(OUT_PDF) as pdf:
    pdf.savefig(plot_summary(acc, y_true, y_pred))
    plt.close()

    pdf.savefig(plot_report_table(report_dict, LABELS))
    plt.close()

    pdf.savefig(plot_legend())
    plt.close()

    pdf.savefig(
        plot_cm(
            cm,
            LABELS,
            "Matriz de Confusão — Contagem Absoluta (escala log)",
            fmt_int=True,
            log_scale=True,
        )
    )
    plt.close()

    pdf.savefig(
        plot_cm(
            cm_norm,
            LABELS,
            "Matriz de Confusão — Normalizada por Linha (recall %)",
            fmt_int=False,
            vmin=0,
            vmax=1,
            log_scale=False,
        )
    )
    plt.close()

print(f"PDF gerado: {OUT_PDF}")
