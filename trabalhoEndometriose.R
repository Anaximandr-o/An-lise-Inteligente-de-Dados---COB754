library(dplyr)
library(gtsummary)

dados <- readRDS("dados_endometriose_limpos.rds")
dadosTeste <- readRDS("dados_endometriose_rj_2025.rds")

saveRDS(dados, file = "dados_endometriose_limpos.rds")

# --- Explicando variáveis --- #
# mesmo_municipio: Paciente se manteve no mesmo município para internação? Sim ou não
# ESPEC: Onde a paciente foi internada ou a especialidade principal do tratamento realizado

# ── Tabela 1: Mesmo município, município de residência e hospitais agrupados ──────────────────────
dados %>%
  select(mesmo_municipio, munResNome_Agrupado, CAR_CAT) %>%
  tbl_summary(
    by = CAR_CAT,
    percent = "row",
    statistic = list(
      all_continuous()  ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 2
  ) %>%
  add_p(
    test = list(
      all_continuous()  ~ "wilcox.test",
      all_categorical() ~ "chisq.test"
    )
  ) %>% bold_labels()

dados %>%
  select(Hospital_Agrupado, CAR_CAT) %>%
  tbl_summary(
    by = CAR_CAT,
    percent = "row",
    statistic = list(
      all_continuous()  ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 2
  ) %>%
  add_p(
    test = list(
      all_continuous()  ~ "wilcox.test",
      all_categorical() ~ "chisq.test"
    )
  ) %>% bold_labels()


# ── Tabela 2: Idade, RACA_COR e sexo  ──────────────────────
dados %>%
  select(IDADE, RACA_COR, SEXO, CAR_CAT) %>%
  tbl_summary(
    by = CAR_CAT,
    percent = "row",
    statistic = list(
      all_continuous()  ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 2
  ) %>%
  add_p(
    test = list(
      all_continuous()  ~ "wilcox.test",
      all_categorical() ~ "chisq.test"
    )
  ) %>% bold_labels()



# ── Tabela 3: DIAG_PRINC_label, ESPEC e COMPLEX  ──────────────────────
dados %>%
  select(ESPEC, COMPLEX, CAR_CAT) %>%
  tbl_summary(
    by = CAR_CAT,
    percent = "row",
    statistic = list(
      all_continuous()  ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 2
  ) %>%
  add_p(
    test = list(
      all_continuous()  ~ "wilcox.test",
      all_categorical() ~ "chisq.test"
    )
  ) %>% bold_labels()


dados %>%
  select(DIAG_PRINC_label, CAR_CAT) %>%
  tbl_summary(
    by = CAR_CAT,
    percent = "row",
    statistic = list(
      all_continuous()  ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 2
  ) %>%
  add_p(
    test = list(
      all_continuous()  ~ "wilcox.test",
      all_categorical() ~ "chisq.test"
    )
  ) %>% bold_labels()
