if (!require("devtools")){install.packages("devtools")}
devtools::install_github("trnnick/tsutils")

require(tsutils)

#REGRESS�O

dados <- read.csv("D:/Dropbox/MauricioPessoal/P�s-Gradua��o/PPGESP/Projeto Final/Estudo Python/Bases C�rdio Pulmonar/V3/30-TESTES-REGRESSAO-CSV.csv")

matriz <- as.matrix(dados)

nemenyi(matriz, conf.level = 0.95, plottype = "vline")

nemenyi(matriz, conf.level = 0.95, plottype = "vmcb")

require(PMCMR)

friedman.test(matriz)

posthoc.friedman.nemenyi.test(matriz)

#REGRESS�O XGB

dados <- read.csv("C:/Users/Mauricio/Dropbox/MauricioPessoal/P�s-Gradua��o/PPGESP/Projeto Final/Estudo Python/Bases C�rdio Pulmonar/V3/30-TESTES-REGRESSAO-XGB-CSV.csv")

matriz <- as.matrix(dados)

nemenyi(matriz, conf.level = 0.95, plottype = "vline")

#CLASSIFICACAO

dados <- read.csv("C:/Users/Mauricio/Dropbox/MauricioPessoal/P�s-Gradua��o/PPGESP/Projeto Final/Estudo Python/Bases C�rdio Pulmonar/V3/30-TESTES-CLASSIFICACAO-CSV.csv")

matriz <- as.matrix(dados)

nemenyi(matriz, conf.level = 0.95, plottype = "vline")

#CLASSIFICACAO LIGHTGBM

dados <- read.csv("C:/Users/mauricio.purificacao/Dropbox/MauricioPessoal/P�s-Gradua��o/PPGESP/Projeto Final/Estudo Python/Bases C�rdio Pulmonar/V3/30-TESTES-CLASSIFICACAO-LIGHTGBM-CSV.csv")

matriz <- as.matrix(dados)

nemenyi(matriz, conf.level = 0.95, plottype = "vline")