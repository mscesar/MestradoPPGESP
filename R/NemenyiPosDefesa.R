if (!require("devtools")){install.packages("devtools")}
devtools::install_github("trnnick/tsutils")

require(tsutils)

#REGRESSÃO

dados <- read.csv("D:/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar/V5/30-TESTES-REGRESSAO-CSV-2.csv")

matriz <- as.matrix(dados)

nemenyi(matriz, conf.level = 0.95, plottype = "vline")

nemenyi(matriz, conf.level = 0.95, plottype = "vmcb")

require(PMCMR)

friedman.test(matriz)

posthoc.friedman.nemenyi.test(matriz)

#REGRESSÃO XGB

dados <- read.csv("D:/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar/V5/30-TESTES-REGRESSAO-XGB-CSV.csv")

matriz <- as.matrix(dados)

nemenyi(matriz, conf.level = 0.95, plottype = "vline")

#CLASSIFICACAO

dados <- read.csv("C:/Users/Mauricio/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar/V3/30-TESTES-CLASSIFICACAO-CSV.csv")

matriz <- as.matrix(dados)

nemenyi(matriz, conf.level = 0.95, plottype = "vline")

#CLASSIFICACAO LIGHTGBM

dados <- read.csv("C:/Users/mauricio.purificacao/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar/V3/30-TESTES-CLASSIFICACAO-LIGHTGBM-CSV.csv")

matriz <- as.matrix(dados)

nemenyi(matriz, conf.level = 0.95, plottype = "vline")