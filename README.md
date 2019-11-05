# data-analysis-diabetes-ia

**Curso: Ciência de Dados e Big Data | Disciplina: Machine Learning**

**Aplicação de Machine Learning para Auxílio no Diagnóstico do Diabetes** 

Professor: Hugo de Paula
Alunos:
- Anderson Carvalho
- Angela Mendonça
- Camila Coutinho
- Marina Oliveira
- Thiago Silva

**Introdução**
O diabetes é uma síndrome metabólica de origem múltipla, decorrente da falta de insulina e/ou da incapacidade de a insulina exercer adequadamente seus efeitos, causando um aumento da glicose (açúcar) no sangue. Ao longo do tempo, a diabetes pode provocar danos ao coração, vasos sanguíneos, olhos, rins e nervos.
Para elaboração do trabalho utilizamos um conjunto de dados sobre diabetes extraida do site: http://storm.cis.fordham.edu (http://storm.cis.fordham.edu). A base de dados é do ano de 1988 e retratava dados reais de uma população que vivia perto de Phoenix, Arizona, EUA. Possui um total de 767 registros e 8 classes com atributos relacionados a saúde e idade dos pacientes.

**Problema**
Segundo a OMC (Organização mundial de saúde) 1 em cada 11 pessoas no mundo possui diabetes. Esse número representa um total de 420 milhões de pessoas (dados de 2014). Ainda segunda a OMC, a taxa de incidência de diabetes cresceu mais de 61% nos ultimos 10 anos. Só no Brasil, os números apontam que 16 milhões de pessoas sofrem de diabetes.

**Objetivo**
Com base nas informações do conjunto de dados, o objetivo do trabalho é tentar prever com maior assertividade se uma possoa possui ou não pré disposição para o diabetes. A identificação precoce de diabetes é extremamente importante pois, auxilia o diabético a manter um nível bom de glicose podendo evitar consequências graves como, infarto, derrame cerebral ou cegueira.

**Justificativa para uso dos algorítimos**
A árvore de decisão é uma técnica estatística de treinamento supervisionado utilizada na classificação e previsão dos dados. É um modelo que usa a ideia de dividir para conquistar, em outras palavras, decompõe um problema maior em sub-problemas de ordem mais simples, para de forma recursiva se consiga a resolução do problema como um todo. É uma técnica bastante utilizada em diagnóstico médico e, por isso foi escolhida para ser utilizada nesse trabalho.

O Naive Bayes também é um algorítimo classificador probabilístico muito eficiente e de simples implementação. Apenas com uma pequena quantidade de dados é possível obter classificações com uma boa previsão. Essa técnica desconsidera completamente a correlação entre variáveis e portanto, bem diferente da árvore de decisão. Por esse motivo foi escolhido para ser utilizada nesse trabalho.

**Análise dos resultados**
Neste exercício foram utilizados dois algoritmos, a arvore de decisão e o algoritmo de Naive Bayes. Para o problema de diagnóstico da diabete a arvore de decisão apresentou um melhor resultado alcançando 100% de acurácia contra 65,4% do Naive Bayes. Um ponto importante, é que a tratativa dos dados para ambos os algoritmos foi exatamente a mesma, como vimos ao longo da disciplina se forem aplicados métodos diferentes para cada algoritmo o resultado pode ser afetado, porém isso não foi realizado no trabalho em questão.
