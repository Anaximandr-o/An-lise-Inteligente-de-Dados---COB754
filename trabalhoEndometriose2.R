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




# ============================================================
# K-means (aprendizado não supervisionado) com as variáveis confirmadas pelo Boruta
# ============================================================
library(cluster)
library(factoextra)
library(fpc)

### Preparação dos dados ###

# Mesma matriz da MLP: numéricas padronizadas (média e DP do TREINO) + colunas one-hot.
# O K-means não usa a resposta: o CAR_CAT só entra depois, para validar os grupos.
x_treino_km <- x_treino_mlp
x_teste_km  <- x_teste_mlp

# Distâncias euclidianas entre as observações (silhueta e validações)
dist_treino_km <- dist(x_treino_km)
dist_teste_km  <- dist(x_teste_km)

# O K-means não tem predict(): cada observação nova vai para o grupo do centróide mais próximo
atribuir_grupo <- function(x, centros) {
  dist_centros <- sapply(1:nrow(centros), function(j) {
    rowSums(sweep(x, 2, centros[j, ])^2)
  })
  max.col(-dist_centros, ties.method = "first")
}

### Escolha do número de grupos (k) e métricas de avaliação ###

grid_km <- data.frame(k = 2:10)

grid_km$sse      <- NA_real_ # Soma dos Erros Quadráticos
grid_km$r2       <- NA_real_ # Soma dos quadrados entre grupos / total
grid_km$ch       <- NA_real_ # Soma dos quadrados entre grupos / dentro
grid_km$silhueta <- NA_real_ # Distância do próprio grupo e distância dos vizinhos

for (i in 1:nrow(grid_km)) {
  set.seed(100)
  km_i <- kmeans(
    x_treino_km,
    centers = grid_km$k[i],
    nstart = 25,
    iter.max = 100
  )
  sil_i <- silhouette(km_i$cluster, dist_treino_km)
  
  grid_km$sse[i]      <- km_i$tot.withinss
  grid_km$r2[i]       <- km_i$betweenss / km_i$totss
  grid_km$ch[i]       <- (km_i$betweenss / (grid_km$k[i] - 1)) /
    (km_i$tot.withinss / (nrow(x_treino_km) - grid_km$k[i]))
  grid_km$silhueta[i] <- mean(sil_i[, "sil_width"])
}

# Valores de k ordenados pela silhueta média
grid_km[order(-grid_km$silhueta), ]

# k escolhido: maior silhueta média
melhor_km <- grid_km[which.max(grid_km$silhueta), ]
melhor_km

### Gráficos para escolha de k ###

# Cotovelo (screeplot)
ggplot(grid_km, aes(x = k, y = sse)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 3) +
  geom_vline(xintercept = melhor_km$k, linetype = "dashed") +
  scale_x_continuous(breaks = grid_km$k) +
  labs(
    title = "Método do cotovelo - K-means",
    subtitle = "Linha tracejada = maior silhueta média",
    x = "Número de grupos (k)",
    y = "Soma de quadrados dentro dos grupos (SSE)"
  ) +
  theme_minimal(base_size = 13)

# Silhueta média por k
ggplot(grid_km, aes(x = k, y = silhueta)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 3) +
  geom_vline(xintercept = melhor_km$k, linetype = "dashed") +
  scale_x_continuous(breaks = grid_km$k) +
  labs(
    title = "Silhueta média - K-means",
    subtitle = "Linha tracejada = maior silhueta média",
    x = "Número de grupos (k)",
    y = "Largura média da silhueta"
  ) +
  theme_minimal(base_size = 13)

### Modelo final ###

k_final <- 9

set.seed(100)
modelo_km <- kmeans(
  x_treino_km,
  centers = k_final,
  nstart = 25,
  iter.max = 100
)

# Tamanho dos grupos, SSE dentro de cada grupo e R² (SSE entre grupos / SSE total)
modelo_km$size
modelo_km$withinss
modelo_km$betweenss / modelo_km$totss

# Grupo de cada observação (no teste: centróide mais próximo, calculado no treino)
grupo_treino_km <- modelo_km$cluster
grupo_teste_km  <- atribuir_grupo(x_teste_km, modelo_km$centers)

table(grupo_treino_km)
table(grupo_teste_km)

### Visualização dos grupos ###

# Projeção nas duas primeiras componentes principais.
# Centraliza sem reescalar (o stand = FALSE do fviz_cluster também desliga a centralização)
x_treino_km_c <- scale(x_treino_km, scale = FALSE)

fviz_cluster(
  modelo_km,
  data = x_treino_km_c,
  stand = FALSE,
  geom = "point",
  palette = "jco",
  main = "Grupos do K-means (PCA)",
  ggtheme = theme_minimal(base_size = 13)
)

# Mesma projeção, colorida pela classe real
fviz_cluster(
  list(data = x_treino_km_c, cluster = treino$CAR_CAT),
  stand = FALSE,
  geom = "point",
  palette = "jco",
  main = "Classe real (CAR_CAT) na mesma projeção",
  ggtheme = theme_minimal(base_size = 13)
)

# Colunas que mais pesam em cada eixo (Dim1 = PC1, Dim2 = PC2)
pca_km <- prcomp(x_treino_km)
cargas_km <- pca_km$rotation[, 1:2]

cargas_km[order(-abs(cargas_km[, "PC1"]))[1:8], ]
cargas_km[order(-abs(cargas_km[, "PC2"]))[1:8], ]

# Silhueta de cada observação
sil_km <- silhouette(grupo_treino_km, dist_treino_km)

fviz_silhouette(
  sil_km,
  palette = "jco",
  ggtheme = theme_minimal(base_size = 13)
)

# Observações com silhueta negativa (possivelmente no grupo errado)
sum(sil_km[, "sil_width"] < 0)

### Validação interna e externa ###

# cluster.stats (pacote fpc):
# interna = silhueta, Dunn, razão within/between (wb.ratio) e Calinski-Harabasz (ch)
# externa = Rand ajustado e VI (Variation of Information) contra a classe real
stats_treino_km <- cluster.stats(
  dist_treino_km,
  grupo_treino_km,
  alt.clustering = as.integer(treino$CAR_CAT)
)

stats_teste_km <- cluster.stats(
  dist_teste_km,
  grupo_teste_km,
  alt.clustering = as.integer(teste$CAR_CAT)
)

metricas_km <- data.frame(
  Base = c("Treino", "Teste"),
  Silhueta = c(stats_treino_km$avg.silwidth, stats_teste_km$avg.silwidth),
  Dunn = c(stats_treino_km$dunn, stats_teste_km$dunn),
  Razao_WB = c(stats_treino_km$wb.ratio, stats_teste_km$wb.ratio),
  CH = c(stats_treino_km$ch, stats_teste_km$ch),
  Rand_ajustado = c(stats_treino_km$corrected.rand, stats_teste_km$corrected.rand),
  VI = c(stats_treino_km$vi, stats_teste_km$vi)
)

metricas_km

# Grupos x classe real
table(Grupo = grupo_treino_km, CAR_CAT = treino$CAR_CAT)
table(Grupo = grupo_teste_km, CAR_CAT = teste$CAR_CAT)

### Perfil dos grupos (treino) ###

perfil_km <- treino[, vars_boruta, drop = FALSE]
perfil_km$CAR_CAT <- treino$CAR_CAT
perfil_km$Grupo <- factor(grupo_treino_km)

tabela_perfil_km <- tbl_summary(
  perfil_km,
  by = Grupo,
  statistic = all_continuous() ~ "{mean} ({sd})"
)
tabela_perfil_km <- add_overall(tabela_perfil_km)
tabela_perfil_km

### Predições e desempenho ###

# Escore de cada grupo = proporção de Urgencia no TREINO.
# Cada observação do teste recebe o escore do grupo do seu centróide mais próximo.
prop_urg_km <- tapply(
  treino$CAR_CAT == "Urgencia",
  factor(grupo_treino_km, levels = 1:k_final),
  mean
)
prop_urg_km

prob_teste_km <- as.numeric(prop_urg_km[grupo_teste_km])

# % de Urgencia em cada grupo (treino e teste)
prop_urg_teste_km <- tapply(
  teste$CAR_CAT == "Urgencia",
  factor(grupo_teste_km, levels = 1:k_final),
  mean
)

df_urg_km <- data.frame(
  Grupo = factor(rep(1:k_final, times = 2)),
  Base = factor(rep(c("Treino", "Teste"), each = k_final), levels = c("Treino", "Teste")),
  Urgencia = 100 * c(as.numeric(prop_urg_km), as.numeric(prop_urg_teste_km))
)

ggplot(df_urg_km, aes(x = Grupo, y = Urgencia, fill = Base)) +
  geom_col(position = "dodge") +
  geom_text(
    aes(label = round(Urgencia, 1)),
    position = position_dodge(width = 0.9),
    vjust = -0.4
  ) +
  geom_hline(
    yintercept = 100 * mean(treino$CAR_CAT == "Urgencia"),
    linetype = "dashed",
    color = "gray40"
  ) +
  labs(
    title = "% de Urgência por grupo - K-means",
    subtitle = "Linha tracejada = % de Urgência no treino inteiro",
    x = "Grupo",
    y = "% de Urgência"
  ) +
  theme_minimal(base_size = 13)

# ROC e AUC (escore alto = mais Urgencia)
roc_teste_km <- roc(
  teste$CAR_CAT,
  prob_teste_km,
  levels = c("Eletiva", "Urgencia"),
  direction = "<",
  quiet = TRUE
)

auc_teste_km <- auc(roc_teste_km)
auc_teste_km

# IC95% da AUC
ci_auc_km <- ci.auc(roc_teste_km)
ci_auc_km

# Melhor ponto de corte (Índice de Youden)
melhor_corte_km <- coords(
  roc_teste_km,
  x = "best",
  best.method = "youden",
  ret = c("threshold", "specificity", "sensitivity")
)

melhor_corte_km
corte_otimo_km <- as.numeric(melhor_corte_km$threshold[1])

classe_teste_km <- factor(
  ifelse(prob_teste_km >= corte_otimo_km, "Urgencia", "Eletiva"),
  levels = c("Eletiva", "Urgencia")
)

# Matriz de confusão
confusionMatrix(
  classe_teste_km,
  teste$CAR_CAT,
  positive = "Urgencia"
)

### Curva ROC ###

ggroc(roc_teste_km, linewidth = 1.2) +
  geom_abline(
    slope = 1,
    intercept = 1,
    linetype = "dashed",
    color = "gray"
  ) +
  # Adiciona o ponto de corte ótimo no gráfico
  geom_point(
    aes(x = melhor_corte_km$specificity[1], y = melhor_corte_km$sensitivity[1]),
    color = "red",
    size = 4
  ) +
  annotate(
    "text",
    x = melhor_corte_km$specificity[1] - 0.05,
    y = melhor_corte_km$sensitivity[1] - 0.05,
    label = paste("Corte:", round(corte_otimo_km, 2)),
    color = "red",
    size = 4
  ) +
  labs(
    title = "Curva ROC - K-means",
    subtitle = paste0(
      "AUC = ", round(as.numeric(auc_teste_km), 3)
    ),
    x = "Especificidade",
    y = "Sensibilidade"
  ) +
  theme_minimal(base_size = 13)
