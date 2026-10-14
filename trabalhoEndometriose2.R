library(dplyr)
library(gtsummary)
library(pROC)
library(rsample)
library(caret)
library(Boruta)
library(xgboost)

dados <- readRDS("dados_endometriose_limpos.rds")

# Divisão treino/teste
set.seed(100)
divisao <- initial_split(
  dados,
  prop = 0.70,
  strata = CAR_CAT
)
treino <- training(divisao)
teste  <- testing(divisao)

# Define "Urgencia" como a classe principal (nível 1)
treino$CAR_CAT <- relevel(treino$CAR_CAT, ref = "Urgencia")

form_rf <- CAR_CAT ~ IDADE + RACA_COR + NUM_FILHOS + SEXO + DIAG_PRINC +
  ESPEC + COMPLEX + mesmo_municipio + Hospital_Agrupado + munResNome_Agrupado

# Importância por permutação via randomForest (mesmo pacote do RF acima)
set.seed(100)
boruta_resultado <- Boruta(
  form_rf,
  data = treino,
  getImp = getImpLegacyRfZ,
  maxRuns = 100,
  doTrace = 1
)

boruta_resultado

# Estatísticas de importância e decisão para cada variável
attStats(boruta_resultado)

# Variáveis "Tentative" são resolvidas pela mediana da importância
if (any(boruta_resultado$finalDecision == "Tentative")) {
  boruta_final <- TentativeRoughFix(boruta_resultado)
} else {
  boruta_final <- boruta_resultado
}

# Gráfico: verde = confirmada, amarelo = tentativa, vermelho = rejeitada, azul = sombra
par(mar = c(10, 4, 2, 1))
plot(boruta_final, las = 2, xlab = "", cex.axis = 0.8)
par(mar = c(5, 4, 4, 2) + 0.1)

# Variáveis confirmadas pelo Boruta
vars_boruta <- getSelectedAttributes(boruta_final, withTentative = FALSE)
vars_boruta

# ============================================================
# XGBoost (com as variáveis confirmadas pelo Boruta)
# ============================================================

# Matriz numérica com one-hot completo dos fatores
fazer_matriz <- function(df) {
  fatores <- names(df)[sapply(df, is.factor)]
  mat <- model.matrix(
    ~ . - 1,
    data = df,
    contrasts.arg = lapply(df[fatores], contrasts, contrasts = FALSE)
  )
  colnames(mat) <- make.names(colnames(mat), unique = TRUE)
  mat
}

x_treino <- fazer_matriz(treino[, vars_boruta, drop = FALSE])
x_teste  <- fazer_matriz(teste[, vars_boruta, drop = FALSE])

# Resposta: 1 = Urgencia, 0 = Eletiva
y_treino <- as.numeric(treino$CAR_CAT == "Urgencia")
y_teste  <- as.numeric(teste$CAR_CAT == "Urgencia")

dtreino <- xgb.DMatrix(data = x_treino, label = y_treino)
dteste  <- xgb.DMatrix(data = x_teste, label = y_teste)

# Validação-cruzada 10-fold (mesmos folds para todas as combinações)
set.seed(100)
folds_cv <- createFolds(treino$CAR_CAT, k = 10)

# Grade de hiperparâmetros
grid_xgb <- expand.grid(
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1),
  min_child_weight = c(1, 5),
  subsample = 0.8,
  colsample_bytree = 0.8
)

grid_xgb$nrounds <- NA_integer_
grid_xgb$auc_cv  <- NA_real_

for (i in 1:nrow(grid_xgb)) {
  cv_xgb <- xgb.cv(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = grid_xgb$max_depth[i],
      eta = grid_xgb$eta[i],
      min_child_weight = grid_xgb$min_child_weight[i],
      subsample = grid_xgb$subsample[i],
      colsample_bytree = grid_xgb$colsample_bytree[i],
      seed = 100
    ),
    data = dtreino,
    nrounds = 1000,
    nfold = 10,
    folds = folds_cv,
    early_stopping_rounds = 30,
    verbose = FALSE
  )
  
  log_cv <- cv_xgb$evaluation_log
  grid_xgb$nrounds[i] <- log_cv$iter[which.max(log_cv$test_auc_mean)]
  grid_xgb$auc_cv[i]  <- max(log_cv$test_auc_mean)
}

# Combinações ordenadas pela AUC da validação cruzada
grid_xgb[order(-grid_xgb$auc_cv), ]

melhor_xgb <- grid_xgb[which.max(grid_xgb$auc_cv), ]
melhor_xgb

### Modelo final e importância das variáveis ###

modelo_xgb <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = melhor_xgb$max_depth,
    eta = melhor_xgb$eta,
    min_child_weight = melhor_xgb$min_child_weight,
    subsample = melhor_xgb$subsample,
    colsample_bytree = melhor_xgb$colsample_bytree,
    seed = 100
  ),
  data = dtreino,
  nrounds = melhor_xgb$nrounds,
  verbose = 0
)

# Importância das variáveis (Gain, Cover e Frequency)
importancia_xgb <- xgb.importance(model = modelo_xgb)
importancia_xgb

xgb.plot.importance(importancia_xgb, top_n = 15)

### Predições e desempenho ###

# Probs previstas no teste
prob_teste_xgb <- predict(modelo_xgb, dteste)

# ROC e AUC
roc_teste_xgb <- roc(
  teste$CAR_CAT,
  prob_teste_xgb,
  levels = c("Eletiva", "Urgencia"),
  quiet = TRUE
)

auc_teste_xgb <- auc(roc_teste_xgb)
auc_teste_xgb

# IC95% da AUC
ci_auc_xgb <- ci.auc(roc_teste_xgb)
ci_auc_xgb

# Melhor ponto de corte (Índice de Youden)
melhor_corte_xgb <- coords(
  roc_teste_xgb,
  x = "best",
  best.method = "youden",
  ret = c("threshold", "specificity", "sensitivity")
)

melhor_corte_xgb
corte_otimo_xgb <- as.numeric(melhor_corte_xgb$threshold[1])

classe_teste_xgb <- factor(
  ifelse(prob_teste_xgb >= corte_otimo_xgb, "Urgencia", "Eletiva"),
  levels = c("Eletiva", "Urgencia")
)

# Matriz de confusão
confusionMatrix(
  classe_teste_xgb,
  teste$CAR_CAT,
  positive = "Urgencia"
)

### Curva ROC ###

ggroc(roc_teste_xgb, linewidth = 1.2) +
  geom_abline(
    slope = 1,
    intercept = 1,
    linetype = "dashed",
    color = "gray"
  ) +
  # Adiciona o ponto de corte ótimo no gráfico
  geom_point(
    aes(x = melhor_corte_xgb$specificity[1], y = melhor_corte_xgb$sensitivity[1]),
    color = "red",
    size = 4
  ) +
  annotate(
    "text",
    x = melhor_corte_xgb$specificity[1] - 0.05,
    y = melhor_corte_xgb$sensitivity[1] - 0.05,
    label = paste("Corte:", round(corte_otimo_xgb, 2)),
    color = "red",
    size = 4
  ) +
  labs(
    title = "Curva ROC - XGBoost",
    subtitle = paste0(
      "AUC = ", round(as.numeric(auc_teste_xgb), 3)
    ),
    x = "Especificidade",
    y = "Sensibilidade"
  ) +
  theme_minimal(base_size = 13)
