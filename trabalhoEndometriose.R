library(dplyr)
library(gtsummary)
library(pROC)

dados <- readRDS("dados_endometriose_limpos.rds")


# ── Regressão Logística ──────────────────────
# Transformando a variável X em 1 e 0
dados <- dados %>%
  mutate(CAR_CAT = ifelse(CAR_CAT == "Urgencia", 1, 0))

# Treinando o modelo de regressão logística
# Fora do modelo por serem a versão não agrupada/redundante de outra coluna:
modelo <- glm(
  CAR_CAT ~ IDADE + SEXO + RACA_COR + NUM_FILHOS +
    mesmo_municipio + munResNome_Agrupado + Hospital_Agrupado +
    ESPEC + COMPLEX + DIAG_PRINC,
  data = dados,
  family = binomial
)

tidy_wald <- function(x, exponentiate = FALSE, conf.level = 0.95, ...) {
  ic <- confint.default(x, level = conf.level)
  if (exponentiate) ic <- exp(ic)
  
  dplyr::bind_cols(
    broom::tidy(x, exponentiate = exponentiate, conf.int = FALSE),
    dplyr::as_tibble(ic, .name_repair = "minimal") %>%
      rlang::set_names(c("conf.low", "conf.high"))
  )
}

# Probabilidades previstas pelo modelo
prob_previstas <- predict(modelo, type = "response")

# Curva ROC e AUC
roc_modelo <- roc(dados$CAR_CAT, prob_previstas)
auc(roc_modelo)

# IC 95% do AUC (opcional, mas fortalece o artigo)
ci.auc(roc_modelo)


conjunto1 <- c("IDADE", "SEXO", "RACA_COR", "NUM_FILHOS")
conjunto2 <- c("DIAG_PRINC", "ESPEC", "COMPLEX")
conjunto3 <- c("mesmo_municipio", "munResNome_Agrupado")
conjunto4 <- c("Hospital_Agrupado")

tabela1 <- tbl_regression(modelo, include = all_of(conjunto1), exponentiate = TRUE, tidy_fun = tidy_wald)
tabela2 <- tbl_regression(modelo, include = all_of(conjunto2), exponentiate = TRUE, tidy_fun = tidy_wald)
tabela3 <- tbl_regression(modelo, include = all_of(conjunto3), exponentiate = TRUE, tidy_fun = tidy_wald)
tabela4 <- tbl_regression(modelo, include = all_of(conjunto4), exponentiate = TRUE, tidy_fun = tidy_wald)

tabela1
tabela2
tabela3
tabela4