

# Modelo de Classificação de Músicas do Spotify

## 1\. Objetivo do Projeto

Este projeto de Data Science tem como objetivo desenvolver um modelo de **Machine Learning** capaz de classificar músicas do Spotify como **"agitadas"** ou **"lentas"** com base em suas características sonoras (*features*). A análise visa aplicar um fluxo de trabalho completo, desde o pré-processamento dos dados até a otimização de hiperparâmetros e a avaliação de performance do modelo final.

-----

## 2\. Metodologia

O notebook `modelo.ipynb` documenta todo o processo de desenvolvimento, dividido nas seguintes etapas:

### 2.1. Carga e Preparação dos Dados

  * **Fonte de Dados:** O projeto utiliza um conjunto de dados extraído do Spotify, contido em um arquivo `.xlsx`.
  * **Divisão:** Os dados foram divididos em conjuntos de treino e teste (`train_test_split`) para permitir o treinamento e a posterior avaliação imparcial do modelo.

### 2.2. Pré-Processamento e Engenharia de Features

Uma etapa crucial para garantir a eficácia do modelo.

  * **Tratamento de Variáveis Categóricas:**
    A feature `track_genre` (textual) foi transformada em formato numérico utilizando **One-Hot Encoding** (via `pandas.get_dummies`).

    > *Motivo:* Esta técnica cria novas colunas binárias para cada gênero, evitando a relação de ordem artificial que métodos como `LabelEncoder` poderiam introduzir.

  * **Escalonamento de Features (Feature Scaling):**
    As features numéricas foram normalizadas utilizando o `MinMaxScaler` do Scikit-learn, colocando todas na escala entre 0 e 1.

    > *Importante:* O `scaler` foi treinado **apenas com os dados de treino** (`.fit(X_train)`) e aplicado para transformar tanto treino quanto teste (`.transform()`), evitando assim o vazamento de dados (*data leakage*).

### 2.3. Modelagem e Otimização

  * **Modelo Base:** `RandomForestClassifier` (Ensemble Learning), escolhido por sua robustez e capacidade de lidar com relações não-lineares.
  * **Otimização (Grid Search):** Foi utilizado o `GridSearchCV` com validação cruzada (`cv=5`) para testar combinações de:
      * `n_estimators`: Número de árvores na floresta.
      * `max_depth`: Profundidade máxima de cada árvore.
  * **Métrica de Sucesso:** O modelo foi otimizado buscando o melhor **F1-Score** (média harmônica entre precisão e recall), ideal para este tipo de classificação.

### 2.4. Avaliação do Modelo

O melhor modelo foi validado com dados de teste inéditos, utilizando as métricas:

  * Acurácia (Accuracy)
  * Precisão (Precision)
  * Recall (Revocação)
  * F1-Score
  * **Matriz de Confusão:** Para visualização dos acertos e erros por classe.

### 2.5. Previsão em Novos Dados

O pipeline foi validado realizando previsões no arquivo `novos_dados.xlsx`. Foi garantida a aplicação dos **mesmos passos de pré-processamento** (One-Hot Encoding e o `scaler` já treinado) antes da inferência.

-----

## 3\. Tecnologias Utilizadas

  * **Python 3.x**
  * **Pandas:** Manipulação e análise de dados.
  * **NumPy:** Operações numéricas.
  * **Scikit-learn:** Pré-processamento, modelagem (`RandomForest`), otimização (`GridSearchCV`) e métricas.
  * **Jupyter Notebook:** Ambiente de desenvolvimento e documentação.

-----
