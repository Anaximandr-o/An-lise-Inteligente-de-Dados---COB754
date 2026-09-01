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
  prop = 0.70,
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

# Encontrar o melhor ponto de corte baseado no Índice de Youden (maximiza Sensibilidade + Especificidade)
melhor_corte <- coords(
  roc_teste, 
  x = "best", 
  best.method = "youden", 
  ret = c("threshold", "specificity", "sensitivity")
)

print(melhor_corte)

# Extrair apenas o valor numérico do threshold (caso retorne multiplos empates, pegamos o primeiro)
corte_otimo <- as.numeric(melhor_corte$threshold[1])

# Aplicar o novo ponto de corte nas probabilidades previstas
classe_teste_otimizada <- ifelse(
  prob_teste >= corte_otimo,
  "Urgencia",
  "Eletiva"
)

classe_teste_otimizada <- factor(
  classe_teste_otimizada,
  levels = c("Eletiva", "Urgencia")
)

# Gerar a nova matriz de confusão e conferir as métricas
matriz_otimizada <- confusionMatrix(
  classe_teste_otimizada,
  teste$CAR_CAT,
  positive = "Urgencia"
)

matriz_otimizada

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
    intercept = 1, # Ajuste para 1 se estiver usando o eixo padrão do ggroc
    linetype = "dashed",
    color = "gray"
  ) +
  # Adiciona o ponto de corte ótimo no gráfico
  geom_point(
    aes(x = melhor_corte$specificity[1], y = melhor_corte$sensitivity[1]),
    color = "red", size = 4
  ) +
  annotate(
    "text",
    x = melhor_corte$specificity[1] - 0.05,
    y = melhor_corte$sensitivity[1] - 0.05,
    label = paste("Corte:", round(corte_otimo, 2)),
    color = "red", size = 4
  ) +
  labs(
    title = "Curva ROC - Regressão Logística",
    subtitle = paste0(
      "AUC = ",
      round(as.numeric(auc_teste), 3)
    ),
    x = "Especificidade", 
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

# ============================================================
# Random Forest
# ============================================================
library(randomForest)
library(ggplot2)

form_rf <- CAR_CAT ~ IDADE + RACA_COR + NUM_FILHOS + SEXO + DIAG_PRINC +
  ESPEC + COMPLEX + mesmo_municipio + Hospital_Agrupado + munResNome_Agrupado

# Ajustar manualmente mtry e ntree
customRF <- list(type = "Classification",
                 library = "randomForest",
                 loop = NULL)

customRF$parameters <- data.frame(parameter = c("mtry", "ntree"),
                                  class = rep("numeric", 2),
                                  label = c("mtry", "ntree"))

customRF$grid <- function(x, y, len = NULL, search = "grid") {}

customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y,
               mtry = param$mtry,
               ntree = param$ntree)
}

customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)

customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")

customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# Validação-cruzada 10-fold
ctrl <- trainControl(method = "cv",
                     number = 10,
                     allowParallel = T)

grid <- expand.grid(.mtry = c(1:7),
                    .ntree = c(500, 1000, 1500))

rfFit <- train(form_rf,
               method = customRF,
               tuneGrid = grid,
               trControl = ctrl,
               metric = "Accuracy",
               data = treino)
rfFit
plot(rfFit)

### Modelo final e importância das variáveis ###

rf <- randomForest(form_rf, data = treino,
                   importance = T,
                   mtry = rfFit$bestTune$mtry,
                   ntree = rfFit$bestTune$ntree)
rf

plot(rf) # erro OOB
legend("topright", colnames(rf$err.rate),
       col = 1:3,
       cex = 0.8,
       fill = 1:3)

# MeanDecreaseAccuracy: permutação
importance(rf, type = 1)

# MeanDecreaseGini: diminuição total nas impurezas do nó
importance(rf, type = 2)

varImpPlot(rf, sort = T)
varUsed(rf, count = T)

### Predições e desempenho ###

predrf <- predict(rf, teste, type = "prob")

roc_teste_rf <- roc(teste$CAR_CAT, predrf[, "Urgencia"])
auc_teste_rf <- auc(roc_teste_rf)

melhor_corte_rf <- coords(roc_teste_rf, "best",
                          ret = c("threshold", "specificity", "sensitivity"))
corte_otimo_rf <- melhor_corte_rf$threshold[1]

resultrf <- as.factor(ifelse(predrf[, "Urgencia"] > corte_otimo_rf,
                             "Urgencia", "Eletiva"))
confusionMatrix(resultrf, teste$CAR_CAT, positive = "Urgencia")

### Curva ROC ###

ggroc(roc_teste_rf, linewidth = 1.2) +
  geom_abline(
    slope = 1, intercept = 1, # Ajuste para 1 se estiver usando o eixo padrão do ggroc
    linetype = "dashed",
    color = "gray"
  ) +
  # Adiciona o ponto de corte ótimo no gráfico
  geom_point(
    aes(x = melhor_corte_rf$specificity[1], y = melhor_corte_rf$sensitivity[1]),
    color = "red",
    size = 4
  ) +
  annotate(
    "text",
    x = melhor_corte_rf$specificity[1] - 0.05,
    y = melhor_corte_rf$sensitivity[1] - 0.05,
    label = paste("Corte:", round(corte_otimo_rf, 2)),
    color = "red",
    size = 4
  ) +
  labs(
    title = "Curva ROC - Random Forest",
    subtitle = paste0(
      "AUC = ", round(as.numeric(auc_teste_rf), 3)
    ),
    x = "Especificidade",
    y = "Sensibilidade"
  ) +
  theme_minimal(base_size = 13)
