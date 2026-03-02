library(dplyr)
library(tidyr)

caminho_arquivo <- "/home/dell/ia-2026.1/ia2025.1/projeto-diagnostico-se/data/dataset_completo.csv"

df <- read.csv(caminho_arquivo, 
               stringsAsFactors = FALSE, 
               check.names = FALSE,
               na.strings = c("", "NA", "NULL"))

colunas_exames <- c(
  "Lactic Dehydrogenase",
  "Mean corpuscular hemoglobin concentration (MCHC)", 
  "Proteina C reativa mg/dL",
  "Leukocytes",
  "Lymphocytes",
  "Platelets",
  "Monocytes",
  "Neutrophils",
  "Red blood Cells"
)

df_final <- df %>%
  select(any_of(c("Patient ID", "SARS-Cov-2 exam result", "Influenza A, rapid test", "Influenza B, rapid test", colunas_exames))) %>%
  slice(1:1000)

cat("Processamento concluído. Registros capturados:", nrow(df_final), "\n")

write.csv(df_final, "/home/dell/ia-2026.1/ia2025.1/projeto-diagnostico-se/data/dataset_filtrado.csv", row.names = FALSE)