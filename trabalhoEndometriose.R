library(dplyr)
library(gtsummary)
library(pROC)
library(rsample)
library(caret)

dados <- readRDS("dados_endometriose_limpos.rds")

# Divisão treino/teste
set.seed(100)
divisao <- initial_split(
  dados,
  prop = 0.80,
  strata = CAR_CAT
)
treino <- training(divisao)
teste  <- testing(divisao)

# Validação cruzada k-fold
controle_cv <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Treinando o modelo de regressão logística
modelo <- train(
  CAR_CAT ~ IDADE + SEXO + RACA_COR +
    mesmo_municipio + munResNome_Agrupado +
    Hospital_Agrupado + ESPEC + DIAG_PRINC,
  data = treino,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = controle_cv
)

modelo

# Probs previstas no teste
prob_teste <- predict(
  modelo,
  newdata = teste,
  type = "prob"
)[, "Urgencia"]

# Classe prevista usando ponto de corte 0,5
classe_teste <- ifelse(
  prob_teste >= 0.50,
  "Urgencia",
  "Eletiva"
)

classe_teste <- factor(
  classe_teste,
  levels = c("Eletiva", "Urgencia")
)

# Matriz de confusão
matriz <- confusionMatrix(
  classe_teste,
  teste$CAR_CAT,
  positive = "Urgencia"
)

matriz

# Métricas
sensibilidade <- matriz$byClass["Sensitivity"]
especificidade <- matriz$byClass["Specificity"]
precisao <- matriz$byClass["Pos Pred Value"]
f1 <- matriz$byClass["F1"]
acuracia_balanceada <- matriz$byClass["Balanced Accuracy"]

sensibilidade
especificidade
precisao
f1
acuracia_balanceada

# ROC e AUC
roc_teste <- roc(
  teste$CAR_CAT,
  prob_teste,
  levels = c("Eletiva", "Urgencia"),
  quiet = TRUE
)

auc_teste <- auc(roc_teste)
auc_teste

# IC95% da AUC
ci_auc <- ci.auc(roc_teste)
ci_auc

# Curva ROC
plot(
  roc_teste,
  main = "Curva ROC - Regressão Logística"
)

# Gráfico
library(ggplot2)
library(pROC)

ggroc(roc_teste, linewidth = 1.2) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Curva ROC - Regressão Logística",
    subtitle = paste0(
      "AUC = ",
      round(as.numeric(auc_teste), 3)
    ),
    x = "1 - Especificidade",
    y = "Sensibilidade"
  ) +
  theme_minimal(base_size = 13)

# Odds Ratios (OR) com IC95% e p-valor
tabela_OR <- tbl_regression(
  modelo$finalModel,
  exponentiate = TRUE
)

tabela_OR

# ============================================================
# KNN - comparação com a regressão logística
# ============================================================

# KNN
set.seed(100)

modelo_knn <- train(
  CAR_CAT ~ IDADE + SEXO + RACA_COR +
    mesmo_municipio + munResNome_Agrupado +
    Hospital_Agrupado + ESPEC + DIAG_PRINC,
  data = treino,
  method = "knn",
  metric = "ROC",
  trControl = controle_cv,
  preProcess = c("center", "scale"),
  tuneLength = 10
)

modelo_knn


# Probabilidades previstas no teste
prob_knn <- predict(
  modelo_knn,
  newdata = teste,
  type = "prob"
)[, "Urgencia"]


# Classe prevista com ponto de corte 0,50
classe_knn <- ifelse(
  prob_knn >= 0.50,
  "Urgencia",
  "Eletiva"
)

classe_knn <- factor(
  classe_knn,
  levels = c("Eletiva", "Urgencia")
)


# Matriz de confusão
matriz_knn <- confusionMatrix(
  classe_knn,
  teste$CAR_CAT,
  positive = "Urgencia"
)

matriz_knn


# Métricas
sensibilidade_knn <- matriz_knn$byClass["Sensitivity"]
especificidade_knn <- matriz_knn$byClass["Specificity"]
precisao_knn <- matriz_knn$byClass["Pos Pred Value"]
f1_knn <- matriz_knn$byClass["F1"]
acuracia_balanceada_knn <- matriz_knn$byClass["Balanced Accuracy"]

sensibilidade_knn
especificidade_knn
precisao_knn
f1_knn
acuracia_balanceada_knn


# ROC e AUC
roc_knn <- roc(
  teste$CAR_CAT,
  prob_knn,
  levels = c("Eletiva", "Urgencia"),
  quiet = TRUE
)

auc_knn <- auc(roc_knn)
auc_knn


# IC95% da AUC
ci_auc_knn <- ci.auc(roc_knn)
ci_auc_knn

