library(dplyr)
library(tidyr)

caminho_arquivo <- "/home/ramoni/Desktop/ia/ia-2025.2/ia2025.1/projeto-diagnostico-se/data/dataset_completo.csv"

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
  select(any_of(c("Patient ID", "SARS-Cov-2 exam result", "Influenza A, rapid test", "Influenza B, rapid test", colunas_exames)))

total_positivos <- sum(df_final$`SARS-Cov-2 exam result` == "positive", na.rm = TRUE)

cat("Total de registros com dados completos encontrados:", nrow(df_final), "\n")
cat("Total de pacientes com COVID-19 positivo:", total_positivos, "\n")

View(df_final)

write.csv(df_final, "/home/ramoni/Desktop/ia/ia-2025.2/ia2025.1/projeto-diagnostico-se/data/dataset_filtrado.csv")
