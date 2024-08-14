# %% [markdown]
# # Textual Analysis

# %%
# imports
from collections import Counter
import pandas as pd

# %%
dataset = pd.read_csv('./Post Cleaning Datasets/dataset_1_no_stopword_lemma.csv')

# %%
# carrega o dataset e exclui a coluna 'sentimento'
datasetText = dataset.copy()
datasetText = datasetText.drop(columns="sentimento", axis=1)

datasetText.head()

# %%
#gambiarra pra transformar a lista de palavras em uma unica string kkkkkkkkkkkkk
datasetText['comentarios'] = datasetText['comentarios'].apply(eval).apply(' '.join)    

datasetText.head()

# %%
# Cria um contador de palavras a fim de averiguar a frequencia da mesma
#results = Counter()

#datasetText['comentarios'].apply(lambda x: results.update(x.split()))

# Cria um dataframe com as palvras e suas frequencias, em ordem decrescente de frequencia e troca o nome das colunas
#datasetWords = pd.DataFrame(results.most_common(), columns =['Words', 'Frequency'])

#print(datasetWords)

# %%
# calcula a frenquencia de cada palavra
#totalFrequency = datasetWords['Frequency'].sum()

# divide cada linha pelo número total de frequencia das palavras 
#datasetWords['Weighted Percentage'] = ((datasetWords['Frequency'].div(totalFrequency))*100)

#datasetWords

# %% [markdown]
# ### Tópicos e biblioteca de palavras similares aos tópicos

# %% [markdown]
# Lista de tópicos e definição de palavras semelhantes
# esta parte do algoritmo tem como objetivo contar quantas palavras semelhantes aos tópicos existem, afim de verificar a importancia do tópico escolhido eo seu peso calculado em todo o dataset
# 
# 

# %% [markdown]
# Identificação dos tópicos e criação dos dicionários de sinonimos

# %%
# Topicos positivo:
# Recebi antes, entrega rápida, entrega no prazo e antes do prazo 
entrega_prazo = ['recebir bem', 
                 'recebir antes',
                 'entrega antes',
                 'entregar antes',
                 'entrega rápido', 
                 'entregar rápido',                 
                 'entrega prazo',                 
                 'entregar prazo', 
                 'entrega dentro',
                 'entregar dentro',
                 'rápido entregar', 
                 'antes prazo', 
                 'prazo antes',
                 'prazo entregar']

produto_qualidadeBoa = ['produto bom',
                        'produto bem',
                        'produto qualidade',
                        'produto excelente',
                        'produto exatamente',
                        'produto perfeito',
                        'produto ótimo',
                        'produto super',
                        'produto maravilhoso',
                        'amei produto',
                        'super produto',
                        'gostar produto',
                        'recomendar produto',
                        'produto bem',
                        'bom produto',
                        'otimo produto',
                        'recomendo produto']

elogio_loja =  ['excelente loja',
                'otima loja',
                'ótima loja',
                'recomendar loja',
                'bom loja',
                'ótimo loja',
                'otimo loja',
                'loja maravilhosa',
                'gostar loja',
                'bom atendimento',
                'atendimento bom',
                'loja boa',
                'loja bom',
                'loja honesto',
                'loja confiável',
                'loja confiavel',
                'loja eficiente',
                'loja atender']


# %%

# Topicos negativos:
entrega_atrasada = ['entrega atrasar',
                    'entrega atraso',
                    'atraso entregar',
                    'atrasar entregar',
                    'atraso produto',
                    'atrasar produto',
                    'produto atraso',
                    'produto atrasar',
                    'produto demorar',
                    'demorar produto',
                    'demorar entrega',
                    'demorar entregar',
                    'entrega demorar',
                    'entregar demorar']

produto_baixa_qualidade = ['produto ruim',
                           'produto péssimo',
                           'ruim produto',
                           'péssimo produto',
                           'produto não original',
                           'produto diferente',
                           'produto fraco',
                           'produto defeito',
                           'produto vir errar',
                           'produto errar',
                           'produto não',
                           'produto inferior',]

produto_errado = ['não recebir',
                  'errar produto',
                  'produto errar',
                  'encomenda errar',
                  'errar encomenda',
                  'entregar errar',
                  'vir errar',
                  'trocar']


# %%

# Topicos neutros:
produto_problema = ['defeito produto',
                    'errar produto',
                    'produto errar',
                    'produto defeito',
                    'produto quebrar',
                    'encomenda errar',
                    'errar encomenda',
                    'entregar errar',
                    'vir errar',
                    'devolver produto']

entrega_demorada = ['ainda não',
                    'ainda nao',
                    'entregar problema',
                    'problema entregar',
                    'encomenda não',
                    'encomenda nao',
                    'encomenda atrasar',
                    'problema envio',
                    'problema correio',
                    'atrasar encomenda']


# %%
datasetText.head()

# %%
def wordFinder(wordlist):

    #padrão criado para buscas as palavras dentro do dataframe
    pattern = rf"{'|'.join(wordlist)}"  
    print(pattern)

    #busca as palavras dentro do dataframe e cria um novo df com palavras e frequencia
    topic = (datasetText[datasetText['comentarios'].str.contains(pattern)])


    wp_topic = topic.count()/datasetText['comentarios'].count()

    print(topic.count())
    print(wp_topic)
    print()

# %%
topicList = [entrega_prazo,
             produto_qualidadeBoa, 
             elogio_loja, 
             entrega_atrasada, 
             produto_baixa_qualidade, 
             produto_errado, 
             produto_problema,
             entrega_demorada]

[wordFinder(i) for i in topicList]


