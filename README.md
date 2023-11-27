# sentiment-analisys-in-digital-marketing
This is my final exam codes and some other files about the use of Sentiment Analysis and Machine Learning models on Digital Marketing.

The final version of the paper will be published by mid december.

I had to learn a lot of new technologies to develop this project, like:

Dataset and text cleaning, removal and transformation using Pandas and NLTK;

NLTK and Spacy library tokenization, Lemmatization and Stemming techniques and concepts;

Learning new SciKit Learn algorithms for applying Machine Learning models;

Learning Lattent Dirilech Allocation and a topic visualization API technique and concept for topic modeling;

The project was based on the work of Belém Barbosa et al., named "Defining content marketing and its influence on online user behavior: a data-driven prescriptive analytics method". (https://doi.org/10.1007/s10479-023-05261-1)

Changes were made to manipulate a portuguese based corpus, as changing the NLP library from TextBlob to NLTK and SpacY, because of TextBlob not having compatibility with Portuguese.


Files description:

olist_order_reviews_dataset.csv: the original dataset, taken from Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_reviews_dataset.csv)

Data Cleaning.ipynb: script for cleaning the dataset and saving the results in "Clean Datasets" folder, Post Cleaning datasets (with tokenization, lemmatization and stemming) were saved in the "Post Cleaning Datasets" folder

Machine Learning.ipynb: script for the Machine Learning algorithms for textual classification saving the results in "ML Results" folder as a .txt files one for the Machine Learning and other for Cross-Validation results

LDA.ipynb: script for the Lattent Dirilech Allocation model, using concepts such as Bag-of-Words and visualization of the results with pyLDAvis, results were saved in the "LDA Results" folder

Textual Analysis.ipynb: script for a simple textual analysis, using similar words found by the topic modeling technique and counting the frequency of them in the whole dataset


