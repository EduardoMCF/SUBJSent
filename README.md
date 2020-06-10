
# SUBJSent
SUBJSent é uma CLI de classificação de subjetividade de sentenças que tem como objetivo disponibilizar diversas estratégias de classificação de subjetividade de forma simples e descomplicada. Atualmente a classificação se limita à sentenças em inglês, mas há planos para expandi-la também para o português.

A CLI tem duas tasks principais: **train** e **predict**. As configurações de ambas podem ser ajustadas no arquivo config.json. 

## Instalação das dependências
[Python 3.6 ou superior](https://www.python.org/downloads/ "Python 3.6 ou superior").

``` $ pip install -r requirements.txt```


## Train
- #### Exemplo
  ```$ python subjsent --task train```
- #### Requisitos
	- **data_path** - 	caminho para um arquivo **csv** com as colunas **text** e **label**, e.g.
  ```
      text, label
      uma sentença subjetiva, 1
      uma sentença objetiva, 0
      ...
  ```
  - **embeddings**
    - **type** - tipo de embeddings que será utilizado pelo modelo, "word" ou "sentence".
  - **model**
    - **type** - tipo de modelo a ser utilizado, "CNN" ou "MLP".
  - **train**
- #### Saída
	Após a execução a seguinte pasta será criada no diretório saved_models:
    
  ```html
  saved_models/
  ├── Model #/
          ├── model.h5  // Arquivo do modelo.
          ├── results.txt // Arquivo com as métricas de teste.
          ├── plots.png // Imagem com os plots das métricas de treino.
  ```

## Predict
- #### Exemplo
```$ python subjsent --task predict```
- #### Requisitos
	- **data_path** - caminho para um arquivo **csv** com a coluna **text**, contendo as sentenças, e.g.
  ```
  text
  sentença 1
  sentença 2
  ...
  ```
	- **model_path** - caminho para um arquivo h5, contendo o modelo que será utilizado durante o predict.
	- **embeddings**
		- **type** - O tipo de embeddings que será utilizado pelo modelo, "word" ou "sentence".

- #### Saída
	Após a execução a seguinte pasta será criada no diretório data_output:
  ```html
  data_output/
  ├── Result #/
          ├── result.csv  // Arquivo contento o valor predito de cada sentença.
  ```

## Configurações
- **Chave** - **Descrição**
<details>
<summary>Clique para mostrar</summary>
	- **data_path** - caminho para o csv de interesse. No caso do train para o csv com as sentenças e com os labels e no caso do predict para o csv com as sentenças.

	- **model_path** - caminho para um arquivo **h5**, contendo o modelo que será utilizado no predict.

	- **embeddings**
		- **type**  - tipo de embedding que será utilizado, "word" ou "sentence".
		- **length** - dimensão dos embeddings.

		- **path** - caminho para os embeddings, que podem estar em formato txt ou bin no caso de word embeddings ou em formato válido para o [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load "hub.load") no caso de sentence embeddings.

		- **binary** - booleano que indica se o arquivo de embeddings é um **.bin** ou não. 
		- **convert_to_w2v** - booleano que indica se é necessária a conversão para o formato de embeddings do word2vec,  caso esteja utilizando o GloVe marque esse campo como true.

	- **model**
		- **type** - tipo de modelo que será utilizado. Atualmente é possível utilizar dois tipos de modelos: "CNN" e "MLP". Só é possível utilizar a CNN com embeddings de palavras e só é possível utilizar o MLP com embeddings de sentença. 

		- **activation** - função de ativação que será utilizada nas camadas da rede neural, deve conter uma string que representa uma [ativação válida do keras](https://keras.io/api/layers/activations/ "ativação válida do keras") .
		- **dense_connections** - array de inteiros com a quantidade de unidades de cada uma das camadas densas.

		- **dropout_rate** - array com floats entre 0 e 1 que representam as taxas de dropout que serão aplicadas no modelo.

		- **num_filters**  - quantidade de filtros que serão aplicados nas camadas de  convolução,  válido apenas para a CNN.

		- **kernel_sizes** - tamanho das mascaras de convolução, válido apenas para a CNN.

		- **metrics** - array com strings que representam as métricas a serem calculadas durante o treinamento,  deve conter strings que representam [metricas válidas do keras](https://keras.io/api/metrics/ "metricas válidas do keras").

		- **optimizer** - string que representa o otimizador utilizado pelo modelo, deve conter uma string que representa um [otimizador válido do keras](https://keras.io/api/optimizers/ "otimizador válido do keras").

		- **learning_rate** - float que representa a taxa de apredizagem que será aplicada ao otimizador.

		- **loss** - string que representa a função de loss que será utilizada no modelo, deve conter uma string que representa uma [função de loss válida do keras](https://keras.io/api/losses/ "função de loss válida do keras").

	- **train**
		- **test_size** - float que representa a porcentagem dos dados que será separada para o teste.

		- **validation_size** - float que representa a porcentagem do conjunto de treino que será separada para validação.

		- **batch_size** - inteiro que representa o tamanho do *batch* utilizado durante o treinamento.

		- **epochs** - inteiro que representa o número de épocas do treinamento.

		- **plot_history** - booleano que indica se as métricas de treino devem ser plotadas ou não.

	- **preprocess**
		- **remove_stopwords** - booleano que indica se as stopwords deve ou não serem removidas durante o preprocessamento, o conjunto de stopwords utilizado é o english do nltk.
</details>

## Mais Informações
Há duas arquiteturas disponíveis para treinamento, uma CNN e um MLP.  Mais detalhes sobre elas podem ser encontrados [aqui](https://github.com/EduardoMCF/SUBJSent/tree/master/subj_sent/models "aqui").

A arquitetura da CNN foi baseada no artigo [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181/ "Convolutional Neural Networks for Sentence Classification") de [Yoon Kim](https://github.com/yoonkim "Yoon Kim"). 

A CNN deve ser utilizada ,obrigatoriamente, em conjunto com word embeddings, enquanto o MLP deve ser utilizado em conjunto com sentence embeddings.

No preprocessamento das sentenças da CNN é feita a tokenização e posteriormente um padding a fim de normalizar o tamanho do input da mesma. O padding é feito com base na sentença com maior quantidade de tokens. 

No MLP o preprocessamento fica a cargo do modelo utilizado para gerar os embeddings, por isso é importante ficar atento para a quantidade máxima de tokens permitida pelo modelo em questão.

A CNN foi testada com [word2vec](https://code.google.com/archive/p/word2vec/ "word2vec") e [GloVe](https://nlp.stanford.edu/projects/glove/ "GloVe"), enquanto que o MLP foi testado com a [versão 5 do Universal Sentence Encoder Large](https://tfhub.dev/google/universal-sentence-encoder-large/5 "versão 5 do Universal Sentence Encoder Large") - que utiliza transformers.
