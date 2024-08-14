# %% [markdown]
# # Latent Dirichlet Allocation

# %% [markdown]
# A técnica de LDA executada por Barbosa, necessita dividir o dataset resultante com pelo sentimento no qual eles foram identificados. Assim a as próximas iterações do algoritmo serão aplicadas em um grupo positivo, negativo e neutro, separadamente, por conta do desbalanceamento do dataset. Se utilizado como um todo, esta parte do experimento estaria enviesada, por conta de haver muito mais comentários positivos do que os outros, o que faria apenas respostas positivas serem mostradas.
# 

# %%
# import

import pandas as pd
import numpy as np

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from nltk.tokenize import word_tokenize

import os

from pprint import pprint

import pyLDAvis.gensim
import pickle 
import pyLDAvis

# number of topics
num_topics = 8


# %%
#carrega o dataset
dataset = pd.read_csv('./Post Cleaning Datasets/dataset_1_no_stopword_stemm.csv')
datasetLDA = dataset.copy()

datasetLDA.head()

# %%
#gambiarra pra transformar a lista de palavras em uma unica string kkkkkkkkkkkkk
datasetLDA['comentarios'] = datasetLDA['comentarios'].apply(eval).apply(' '.join)

datasetLDA.head()

# %%
# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(datasetLDA['comentarios'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()

# %%
# tokeniza a coluna comentários e cria colunas com bigrams e trigrams
datasetLDA['comentarios'] = datasetLDA['comentarios'].apply(word_tokenize)

datasetLDA.head()

# %%
# carrega funções de bigram e trigram
bigram = gensim.models.Phrases(datasetLDA['comentarios'], min_count=2, threshold=100, delimiter='_') # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[datasetLDA['comentarios']], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# %%
# Atualiza o dataset com bigrams e trigrams
datasetLDA['comentarios'] = make_bigrams(datasetLDA['comentarios'])

datasetLDA['comentarios'] = make_trigrams(datasetLDA['comentarios'])

datasetLDA

# %% [markdown]
# Cria um dicionário de cada dataset, depois cria um corpus onde é feito o processo de Term Document Frequency, outra etapa necessária para a aplicação do LDA
# 

# %%
# divide o dataset em 3, positivo, negativo e neutro
grouped = datasetLDA.groupby(datasetLDA.sentimento)

#cria dataframes para cada sentimento
df_positive = grouped.get_group("positive")
df_negative = grouped.get_group("negative")
df_neutral = grouped.get_group("neutral")

# deleta a coluna 'sentimento' pois não é mais necessária para o experimento
df_positive = df_positive.drop('sentimento', axis=1)
df_negative = df_negative.drop('sentimento', axis=1)
df_neutral = df_neutral.drop('sentimento', axis=1)


df_positive.head()

# %% [markdown]
# Criando o Dicionário, Corpus e TDF 

# %%
df_positive_dic = corpora.Dictionary(df_positive['comentarios'])
df_negative_dic = corpora.Dictionary(df_negative['comentarios'])
df_neutral_dic = corpora.Dictionary(df_neutral['comentarios'])

# Cria um corpus de cada dataset
df_positive_corpus = df_positive['comentarios']
df_negative_corpus = df_negative['comentarios']
df_neutral_corpus = df_neutral['comentarios']

# Term Document Frequency de cada corpus
tdf_positive = [df_positive_dic.doc2bow(text) for text in df_positive_corpus]
tdf_negative = [df_negative_dic.doc2bow(text) for text in df_negative_corpus]
tdf_neutral = [df_neutral_dic.doc2bow(text) for text in df_neutral_corpus]


pprint(df_positive_dic)
# View
pprint(tdf_positive)


# %% [markdown]
# # Aplicando Lattent Dirilech Allocation

# %% [markdown]
# LDA Dataset Positivo
# 

# %%
# Build LDA model
lda_model_positive = gensim.models.LdaMulticore(corpus=tdf_positive,
                                                id2word=df_positive_dic,
                                                num_topics=num_topics,
                                                passes=10,
                                                per_word_topics=True)

# Print the Keyword in the n topics
pprint(lda_model_positive.print_topics())
doc_lda = lda_model_positive[tdf_positive]

# %% [markdown]
# LDA Dataset Negativo

# %%
# Build LDA model
lda_model_negative = gensim.models.LdaMulticore(corpus=tdf_negative,
                                                id2word=df_negative_dic,
                                                num_topics=num_topics,
                                                passes=10,
                                                per_word_topics=True)

# Print the Keyword in the n topics
pprint(lda_model_negative.print_topics())
doc_lda = lda_model_negative[tdf_negative]

# %% [markdown]
# LDA Dataset Neutro

# %%
# Build LDA model
lda_model_neutral = gensim.models.LdaMulticore(corpus=tdf_neutral,
                                                id2word=df_neutral_dic,
                                                num_topics=num_topics,
                                                passes=10,
                                                per_word_topics=True)

# Print the Keyword in the n topics
pprint(lda_model_neutral.print_topics())
doc_lda = lda_model_neutral[tdf_neutral]

# %%
#Calculos das métricas da LDA

# POSITIVO
print()
print('Positive')  
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model_positive, texts=df_positive['comentarios'], dictionary=df_positive_dic, coherence='c_v')
coherence_lda_positive = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda_positive)

coherence_model_ldaTeste = CoherenceModel(model=lda_model_positive, texts=df_positive['comentarios'], dictionary=df_positive_dic, coherence='u_mass')
coherence_lda_positiveTeste = coherence_model_ldaTeste.get_coherence()
print('\nCoherence Score u mass: ', coherence_lda_positiveTeste)


#NEGATIVE
print()
print('Negative')  

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model_negative, texts=df_negative['comentarios'], dictionary=df_negative_dic, coherence='c_v')
coherence_lda_negativo = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda_negativo)

coherence_model_ldaTeste = CoherenceModel(model=lda_model_negative, texts=df_negative['comentarios'], dictionary=df_negative_dic, coherence='u_mass')
coherence_lda_negativoTeste = coherence_model_ldaTeste.get_coherence()
print('\nCoherence Score u mass: ', coherence_lda_negativoTeste)


#NEUTRAL
print()
print('Neutral')  

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model_neutral, texts=df_neutral['comentarios'], dictionary=df_neutral_dic, coherence='c_v')
coherence_lda_neutral = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda_neutral)

coherence_model_ldaTeste = CoherenceModel(model=lda_model_neutral, texts=df_neutral['comentarios'], dictionary=df_neutral_dic, coherence='u_mass')
coherence_lda_neutralTeste = coherence_model_ldaTeste.get_coherence()
print('\nCoherence Score u mass: ', coherence_lda_neutralTeste)

# %% [markdown]
# Inicialização da visualização do LDA
# 

# %%
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./LDA Results/visualLDA_Positive'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself

if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model_positive, tdf_positive, df_positive_dic)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './LDA Results/visualLDA_Positive'+ str(num_topics) +'.html')
LDAvis_prepared

# %%
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./LDA Results/visualLDA_Negative'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself

if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model_negative, tdf_negative, df_negative_dic)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './LDA Results/visualLDA_Negative'+ str(num_topics) +'.html')
LDAvis_prepared

# %%
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./LDA Results/visualLDA_Neutral'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself

if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model_neutral, tdf_neutral, df_neutral_dic)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './LDA Results/visualLDA_Neutral'+ str(num_topics) +'.html')
LDAvis_prepared


