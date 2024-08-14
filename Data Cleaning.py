# %% [markdown]
# # Pré-processamento do Dataset

# %%
#imports 

import re
import pandas as pd

import spacy

import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopword = set(stopwords.words('portuguese'))
stopword.discard("não")
stopword.discard("nem")
stopword.discard("mas")
stemmer = nltk.stem.SnowballStemmer("portuguese")
lemma = spacy.load("pt_core_news_sm")

#nltk.download()


# %% [markdown]
# Definindo funções
# 

# %%
def remove_non_alphabet(text):
    return re.sub(r'[^a-zA-Z_À-ÿ]', ' ', text)

def remove_extra_whitespaces(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def remove_stopwords(text):
    t = [token for token in text if token not in stopword]
    text = ' '.join(t)
    return text

# %% [markdown]
# Carregamento do arquivo .csv inicial e exclusão e troca de nome de colunas

# %%
dataset = pd.read_csv('olist_order_reviews_dataset.csv')
dataset = dataset.drop(dataset.columns[[0,1,3,5,6]], axis=1)
dataset = dataset.rename(columns={'review_score': 'sentimento', 'review_comment_message': 'comentarios'})
dataset['sentimento'] = dataset['sentimento'].astype('int')

dataset.head()

# %% [markdown]
# 
# Removendo linhas sem comentários, linhas com apenas números ou pontuações e linhas com comentários duplicados

# %%
dataset = dataset.dropna()

for index, row in dataset.iterrows():
    if not row['comentarios'].upper().isupper():
        dataset = dataset.drop(index)

dataset = dataset.drop_duplicates(subset=None , keep='first')

dataset.head()

# %% [markdown]
# ### Troca dos valores númericos na coluna "sentimento" para string de acordo com duas regras propostas

# %%
dataset_1 = dataset.copy()
dataset_2 = dataset.copy()

#dataset_1 representa os comentários onde 5 e 4 são positivos, 3 neutro e 2 e 1 negativos
dataset_1['sentimento'] = dataset_1['sentimento'].replace([5, 4], 'positive')
dataset_1['sentimento'] = dataset_1['sentimento'].replace(3, 'neutral')
dataset_1['sentimento'] = dataset_1['sentimento'].replace([2, 1], 'negative')

#dataset_2 representa os comentários onde 5 é positivo, 4, 3 e 2 são neutros e 1 é negativo
dataset_2['sentimento'] = dataset_2['sentimento'].replace(5, 'positive')
dataset_2['sentimento'] = dataset_2['sentimento'].replace([4, 3, 2], 'neutral')
dataset_2['sentimento'] = dataset_2['sentimento'].replace(1, 'negative')

dataset_2.head()

# %% [markdown]
# ### Remoção de Stopwords, espaços em branco, não alfabéticos e lowercase

# %%
dataset_1_no_stopword = dataset_1.copy()
dataset_1_with_stopword = dataset_1.copy()

dataset_2_no_stopword = dataset_2.copy()
dataset_2_with_stopword = dataset_2.copy()

# removing stopwords, extra space
dataset_1_no_stopword['comentarios'] = dataset_1_no_stopword['comentarios'].apply(lambda x: x.lower())
dataset_1_no_stopword['comentarios'] = dataset_1_no_stopword['comentarios'].apply(remove_non_alphabet)
dataset_1_no_stopword['comentarios'] = dataset_1_no_stopword['comentarios'].apply(remove_extra_whitespaces)
dataset_1_no_stopword['comentarios'] = dataset_1_no_stopword['comentarios'].apply(word_tokenize)
dataset_1_no_stopword['comentarios'] = dataset_1_no_stopword['comentarios'].apply(remove_stopwords)

dataset_2_no_stopword['comentarios'] = dataset_2_no_stopword['comentarios'].apply(lambda x: x.lower())
dataset_2_no_stopword['comentarios'] = dataset_2_no_stopword['comentarios'].apply(remove_non_alphabet)
dataset_2_no_stopword['comentarios'] = dataset_2_no_stopword['comentarios'].apply(remove_extra_whitespaces)
dataset_2_no_stopword['comentarios'] = dataset_2_no_stopword['comentarios'].apply(word_tokenize)
dataset_2_no_stopword['comentarios'] = dataset_2_no_stopword['comentarios'].apply(remove_stopwords)


#with stopwords
dataset_1_with_stopword['comentarios'] = dataset_1_with_stopword['comentarios'].apply(lambda x: x.lower())
dataset_1_with_stopword['comentarios'] = dataset_1_with_stopword['comentarios'].apply(remove_non_alphabet)
dataset_1_with_stopword['comentarios'] = dataset_1_with_stopword['comentarios'].apply(remove_extra_whitespaces)

dataset_2_with_stopword['comentarios'] = dataset_2_with_stopword['comentarios'].apply(lambda x: x.lower())
dataset_2_with_stopword['comentarios'] = dataset_2_with_stopword['comentarios'].apply(remove_non_alphabet)
dataset_2_with_stopword['comentarios'] = dataset_2_with_stopword['comentarios'].apply(remove_extra_whitespaces)


# remove células que não não NaN porém estão vazias, como por exemplo, uma célula que tem apenas "espaço", essa remoção deve ser feita após a limpeza
dataset_1_no_stopword = dataset_1_no_stopword[dataset_1_no_stopword['comentarios'].str.strip().astype(bool)]
dataset_2_no_stopword = dataset_2_no_stopword[dataset_2_no_stopword['comentarios'].str.strip().astype(bool)]
dataset_1_with_stopword = dataset_1_with_stopword[dataset_1_with_stopword['comentarios'].str.strip().astype(bool)]
dataset_2_with_stopword = dataset_2_with_stopword[dataset_2_with_stopword['comentarios'].str.strip().astype(bool)]


# reseta o indice por questões de compatibilidade
dataset_1_no_stopword = dataset_1_no_stopword.reset_index().drop(columns='index')
dataset_2_no_stopword = dataset_2_no_stopword.reset_index().drop(columns='index')

dataset_1_with_stopword = dataset_1_with_stopword.reset_index().drop(columns='index')
dataset_2_with_stopword = dataset_2_with_stopword.reset_index().drop(columns='index')


dataset_1_no_stopword.head()

# %% [markdown]
# Salvando os datasets

# %%
dataset_1_no_stopword.to_csv('./Clean Datasets/dataset_1_no_stopword.csv', sep=',', encoding='utf-8', index=False)
dataset_1_with_stopword.to_csv('./Clean Datasets/dataset_1_with_stopword.csv', sep=',', encoding='utf-8', index=False)

dataset_2_no_stopword.to_csv('./Clean Datasets/dataset_2_no_stopword.csv', sep=',', encoding='utf-8', index=False)
dataset_2_with_stopword.to_csv('./Clean Datasets/dataset_2_with_stopword.csv', sep=',', encoding='utf-8', index=False)

# %% [markdown]
# ### Copiando os datasets para lemmatização e stemming
# 

# %%
#Lemma

dataset_1_no_stopword_lemma = dataset_1_no_stopword.copy()
dataset_1_with_stopword_lemma = dataset_1_with_stopword.copy()

dataset_2_no_stopword_lemma = dataset_2_no_stopword.copy()
dataset_2_with_stopword_lemma = dataset_2_with_stopword.copy()

#Stemmer

dataset_1_no_stopword_stemm = dataset_1_no_stopword.copy()
dataset_1_with_stopword_stemm = dataset_1_with_stopword.copy()

dataset_2_no_stopword_stemm = dataset_2_no_stopword.copy()
dataset_2_with_stopword_stemm = dataset_2_with_stopword.copy()

# %% [markdown]
# # Lemmatização dos tokens utilizando SpacY
# 

# %% [markdown]
# a lematização não requer que as palavras estejam em formato de token, porém após a lemmatização os dataset passaram pelo processo de tokenization para a aplicação do Bag of Words posterior
# 

# %%
#dataset_1

dataset_1_no_stopword_lemma['comentarios'] = dataset_1_no_stopword_lemma['comentarios'].apply(lambda x: ' '.join([y.lemma_ for y in lemma(x)]))
dataset_1_no_stopword_lemma['comentarios'] = dataset_1_no_stopword_lemma['comentarios'].apply(word_tokenize)

dataset_1_with_stopword_lemma['comentarios'] = dataset_1_with_stopword_lemma['comentarios'].apply(lambda x: ' '.join([y.lemma_ for y in lemma(x)]))
dataset_1_with_stopword_lemma['comentarios'] = dataset_1_with_stopword_lemma['comentarios'].apply(word_tokenize)


#dataset_2

dataset_2_no_stopword_lemma['comentarios'] = dataset_2_no_stopword_lemma['comentarios'].apply(lambda x: ' '.join([y.lemma_ for y in lemma(x)]))
dataset_2_no_stopword_lemma['comentarios'] = dataset_2_no_stopword_lemma['comentarios'].apply(word_tokenize)

dataset_2_with_stopword_lemma['comentarios'] = dataset_2_with_stopword_lemma['comentarios'].apply(lambda x: ' '.join([y.lemma_ for y in lemma(x)]))
dataset_2_with_stopword_lemma['comentarios'] = dataset_2_with_stopword_lemma['comentarios'].apply(word_tokenize)

dataset_1_no_stopword_lemma.head()

# %% [markdown]
# # Stematização com NLTK

# %% [markdown]
# Para a stematização é necessário que as palavras estejam em formato de lista e pode ser feito através da tokenização dessas palavras
# 

# %%
#dataset_1

dataset_1_no_stopword_stemm['comentarios'] = dataset_1_no_stopword_stemm['comentarios'].apply(word_tokenize)
dataset_1_no_stopword_stemm['comentarios'] = dataset_1_no_stopword_stemm['comentarios'].apply(lambda x: [stemmer.stem(y) for y in x])

dataset_1_with_stopword_stemm['comentarios'] = dataset_1_with_stopword_stemm['comentarios'].apply(word_tokenize)
dataset_1_with_stopword_stemm ['comentarios'] = dataset_1_with_stopword_stemm['comentarios'].apply(lambda x: [stemmer.stem(y) for y in x])


#dataset_2

dataset_2_no_stopword_stemm['comentarios'] = dataset_2_no_stopword_stemm['comentarios'].apply(word_tokenize)
dataset_2_no_stopword_stemm['comentarios'] = dataset_2_no_stopword_stemm['comentarios'].apply(lambda x: [stemmer.stem(y) for y in x])

dataset_2_with_stopword_stemm['comentarios'] = dataset_2_with_stopword_stemm['comentarios'].apply(word_tokenize)
dataset_2_with_stopword_stemm['comentarios'] = dataset_2_with_stopword_stemm['comentarios'].apply(lambda x: [stemmer.stem(y) for y in x])

dataset_1_no_stopword_stemm.head()

# %% [markdown]
# Salvando os datasets
# 

# %%
#dataset_1

dataset_1_no_stopword_lemma.to_csv('./Post Cleaning Datasets/dataset_1_no_stopword_lemma.csv', sep=',', encoding='utf-8', index=False)
dataset_1_with_stopword_lemma.to_csv('./Post Cleaning Datasets/dataset_1_with_stopword_lemma.csv', sep=',', encoding='utf-8', index=False)

dataset_1_no_stopword_stemm.to_csv('./Post Cleaning Datasets/dataset_1_no_stopword_stemm.csv', sep=',', encoding='utf-8', index=False)
dataset_1_with_stopword_stemm.to_csv('./Post Cleaning Datasets/dataset_1_with_stopword_stemm.csv', sep=',', encoding='utf-8', index=False)

#dataset_2

dataset_2_no_stopword_lemma.to_csv('./Post Cleaning Datasets/dataset_1_no_stopword_lemma.csv', sep=',', encoding='utf-8', index=False)
dataset_2_with_stopword_lemma.to_csv('./Post Cleaning Datasets/dataset_1_with_stopword_lemma.csv', sep=',', encoding='utf-8', index=False)

dataset_2_no_stopword_stemm.to_csv('./Post Cleaning Datasets/dataset_1_no_stopword_stemm.csv', sep=',', encoding='utf-8', index=False)
dataset_2_with_stopword_stemm.to_csv('./Post Cleaning Datasets/dataset_1_with_stopword_stemm.csv', sep=',', encoding='utf-8', index=False)



