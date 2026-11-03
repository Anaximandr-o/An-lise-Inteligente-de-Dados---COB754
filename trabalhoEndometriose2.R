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
  max_depth = c(4, 6, 8),
  eta = c(0.01, 0.1, 1),
  subsample = c(0.7, 0.8, 0.9),
  colsample_bytree = c(0.7, 0.8, 0.9)
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




# ============================================================
# MLP (rede neural) com as variáveis confirmadas pelo Boruta
# ============================================================
library(nnet)

### Preparação dos dados ###

# A MLP é sensível à escala: padroniza (média 0, DP 1) só as variáveis
# numéricas, usando a média e o DP do TREINO (também aplicados no teste).
# As colunas one-hot (0/1) ficam como estão.
vars_num <- vars_boruta[sapply(treino[, vars_boruta, drop = FALSE], is.numeric)] # Identifica variáveis numéricas
medias   <- colMeans(x_treino[, vars_num, drop = FALSE])
desvios  <- apply(x_treino[, vars_num, drop = FALSE], 2, sd)

x_treino_mlp <- x_treino
x_teste_mlp  <- x_teste
x_treino_mlp[, vars_num] <- scale(x_treino[, vars_num, drop = FALSE], center = medias, scale = desvios) # Padronização dos dados utilizando média e desvio padrão
x_teste_mlp[, vars_num]  <- scale(x_teste[, vars_num, drop = FALSE], center = medias, scale = desvios) # Padroniza treino e teste separadamente

### Treinamento com validação cruzada ###

# Mesmos 10 folds do XGBoost
indices_treino <- lapply(folds_cv, function(f) setdiff(seq_len(nrow(treino)), f))

controle_mlp <- trainControl(
  method = "cv",
  index = indices_treino,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

grid_mlp <- expand.grid(
  size = c(5, 10, 15), # Qtd de neurônios na camada oculta
  decay = c(0.01, 0.1, 1, 5) # Penalização dos pesos
)

set.seed(100)
cv_mlp <- train(
  x = x_treino_mlp,
  y = treino$CAR_CAT,
  method = "nnet",
  metric = "ROC",
  trControl = controle_mlp,
  tuneGrid = grid_mlp,
  maxit = 500,      # máximo de iterações do treinamento
  MaxNWts = 10000,  # limite de pesos da rede
  trace = FALSE
)

# Combinações ordenadas pela AUC da validação cruzada
cv_mlp$results[order(-cv_mlp$results$ROC), ]

melhor_mlp <- cv_mlp$bestTune
melhor_mlp

# Modelo final (o caret já treina de novo com todo o treino na melhor combinação)
modelo_mlp <- cv_mlp$finalModel
modelo_mlp

### Predições e desempenho ###

# Probs previstas no teste (probabilidade de Urgencia)
prob_teste_mlp <- predict(cv_mlp, x_teste_mlp, type = "prob")[, "Urgencia"]

# ROC e AUC
roc_teste_mlp <- roc(
  teste$CAR_CAT,
  prob_teste_mlp,
  levels = c("Eletiva", "Urgencia"),
  quiet = TRUE
)

auc_teste_mlp <- auc(roc_teste_mlp)
auc_teste_mlp

# IC95% da AUC
ci_auc_mlp <- ci.auc(roc_teste_mlp)
ci_auc_mlp

# Melhor ponto de corte (Índice de Youden)
melhor_corte_mlp <- coords(
  roc_teste_mlp,
  x = "best",
  best.method = "youden",
  ret = c("threshold", "specificity", "sensitivity")
)

melhor_corte_mlp
corte_otimo_mlp <- as.numeric(melhor_corte_mlp$threshold[1])

classe_teste_mlp <- factor(
  ifelse(prob_teste_mlp >= corte_otimo_mlp, "Urgencia", "Eletiva"),
  levels = c("Eletiva", "Urgencia")
)

# Matriz de confusão
confusionMatrix(
  classe_teste_mlp,
  teste$CAR_CAT,
  positive = "Urgencia"
)

### Curva ROC ###

ggroc(roc_teste_mlp, linewidth = 1.2) +
  geom_abline(
    slope = 1,
    intercept = 1,
    linetype = "dashed",
    color = "gray"
  ) +
  # Adiciona o ponto de corte ótimo no gráfico
  geom_point(
    aes(x = melhor_corte_mlp$specificity[1], y = melhor_corte_mlp$sensitivity[1]),
    color = "red",
    size = 4
  ) +
  annotate(
    "text",
    x = melhor_corte_mlp$specificity[1] - 0.05,
    y = melhor_corte_mlp$sensitivity[1] - 0.05,
    label = paste("Corte:", round(corte_otimo_mlp, 2)),
    color = "red",
    size = 4
  ) +
  labs(
    title = "Curva ROC - MLP",
    subtitle = paste0(
      "AUC = ", round(as.numeric(auc_teste_mlp), 3)
    ),
    x = "Especificidade",
    y = "Sensibilidade"
  ) +
  theme_minimal(base_size = 13)
