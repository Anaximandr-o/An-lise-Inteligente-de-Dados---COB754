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

# ============================================================
# Random Forest
# ============================================================
library(randomForest)
library(ggplot2)

# Define "Urgencia" como a classe principal (nível 1)
treino$CAR_CAT <- relevel(treino$CAR_CAT, ref = "Urgencia")

form_rf <- CAR_CAT ~ IDADE + RACA_COR + NUM_FILHOS + SEXO + DIAG_PRINC +
  ESPEC + COMPLEX + mesmo_municipio + Hospital_Agrupado + munResNome_Agrupado

# Ajustar manualmente mtry e ntree
customRF <- list(type = "Classification",
                 library = "randomForest",
                 loop = NULL) # Problema de classificação

customRF$parameters <- data.frame(parameter = c("mtry", "ntree"),
                                  class = rep("numeric", 2),
                                  label = c("mtry", "ntree")) # Utilizando os hiperparâmetros de variáveis por divisão e número de árvores

customRF$grid <- function(x, y, len = NULL, search = "grid") {} # Grade de combinações

customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) { # Hiperparâmetros para utilizar no treinamento
  randomForest(x, y,
               mtry = param$mtry,
               ntree = param$ntree)
}

customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) # Previsões da classe final
  predict(modelFit, newdata)

customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL) # Prob de cada classe
  predict(modelFit, newdata, type = "prob")

customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# Validação-cruzada 10-fold
ctrl <- trainControl(method = "cv",
                     number = 10,
                     allowParallel = T,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

grid <- expand.grid(.mtry = c(3:10),
                    .ntree = c(250, 500, 1000))

rfFit <- train(form_rf,
               method = customRF,
               tuneGrid = grid,
               trControl = ctrl,
               metric = "ROC",
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

# MeanDecreaseAccuracy: importância por permutação
importance(rf, type = 1)

# MeanDecreaseGini: importância por diminuição total nas impurezas do nó
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
