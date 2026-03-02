import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("../../../data/dataset_filtrado.csv", decimal=",")

df.columns = df.columns.str.replace('\xa0', ' ').str.strip()

colunas = [
    'SARS-Cov-2 exam result',
    'Influenza A, rapid test',
    'Influenza B, rapid test',
    'Lactic Dehydrogenase',
    'Mean corpuscular hemoglobin concentration (MCHC)',
    'Proteina C reativa mg/dL',
    'Leukocytes',
    'Lymphocytes',
    'Platelets',
    'Monocytes',
    'Neutrophils',
    'Red blood Cells'
]

df = df[colunas].copy()

df.fillna(df.mean(numeric_only=True), inplace=True)

variaveis_laboratoriais = [
    'Lactic Dehydrogenase',
    'Mean corpuscular hemoglobin concentration (MCHC)',
    'Proteina C reativa mg/dL',
    'Leukocytes',
    'Lymphocytes',
    'Platelets',
    'Monocytes',
    'Neutrophils',
    'Red blood Cells'
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

df["Classe_real"] = df.apply(classe_real, axis=1)

def sistema_especialista(row):

    score_covid = 0
    score_flu_a = 0
    score_flu_b = 0

    if row["Lactic Dehydrogenase"] > 0.5:
        score_covid += 1
    if row["Proteina C reativa mg/dL"] > 0.5:
        score_covid += 1
    if row["Neutrophils"] > 0.5:
        score_covid += 1

    if row["Leukocytes"] < -0.5:
        score_flu_a += 1
    if row["Lymphocytes"] < -0.5:
        score_flu_a += 1

    if row["Monocytes"] > 0.5:
        score_flu_b += 1
    if row["Red blood Cells"] < -0.5:
        score_flu_b += 1

    if score_covid >= 2:
        return "COVID-19"
    elif score_flu_a >= 2:
        return "Influenza A"
    elif score_flu_b >= 2:
        return "Influenza B"
    else:
        return "Saudável"

df["Predicao_SE"] = df.apply(sistema_especialista, axis=1)

print("Acurácia:")
print(accuracy_score(df["Classe_real"], df["Predicao_SE"]))

print("\nRelatório de Classificação:")
print(classification_report(df["Classe_real"], df["Predicao_SE"]))

print("\nMatriz de Confusão:")
print(confusion_matrix(df["Classe_real"], df["Predicao_SE"]))