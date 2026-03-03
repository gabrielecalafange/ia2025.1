import pandas as pd
import numpy as np

# Carregar dataset
df = pd.read_csv("/home/ramoni/Desktop/ia/ia-2025.2/ia2025.1/projeto-diagnostico-se/data/dataset_filtrado.csv", decimal=",")

estatisticas = {}

for coluna in df.columns:
    
    serie = pd.to_numeric(df[coluna], errors="coerce")
    serie = serie.replace([np.inf, -np.inf], np.nan)
    
    # Conta valores válidos
    validos = serie.dropna()
    
    if len(validos) > 0:
        
        media = validos.mean()
        desvio = validos.std()
        
        print(f'{coluna} - média: "{media:.6f}"; desvio "{desvio:.6f}"')