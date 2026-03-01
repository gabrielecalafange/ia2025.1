library(dplyr)
library(tidyr)

df <- read.csv("/home/ramoni/Desktop/ia/ia-2025.2/ia2025.1/projeto-diagnostico-se/data/dataset_completo.csv", stringsAsFactors = FALSE, check.names = FALSE)

colunas_selecionadas <- c(
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

df_filtrado <- df %>%
  select(any_of(c("Patient ID", "SARS-Cov-2 exam result", colunas_selecionadas)))


df_reduzido <- df %>%
 
  select(any_of(colunas_selecionadas)) %>%
  slice(1:1000) %>%
  mutate(across(everything(), as.character)) %>%
  mutate(across(everything(), ~ replace_na(.x, "N/A")))


print(paste("Linhas:", nrow(df_reduzido)))
print(paste("Colunas:", ncol(df_reduzido)))
head(df_reduzido)

write.csv(df_filtrado, "/home/ramoni/Desktop/ia/ia-2025.2/ia2025.1/projeto-diagnostico-se/data/dataset_filtrado.csv", row.names = FALSE)