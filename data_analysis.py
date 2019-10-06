import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

def get_polarity_over_time(reviews_df, window=720):
    #window = number of days
    current_date = reviews_df.date.min()
    end_date = reviews_df.date.max()
    dates = []
    polarity_average = []
    window_start = current_date - pd.Timedelta(int(window/2),unit='D')
    window_end = current_date + pd.Timedelta(int(window/2),unit='D')

    time_delta = pd.Timedelta(30,unit='D') #How often to calculate average
    
    d = []
    
    while current_date < end_date:
        polarity_average = reviews_df.polarity[(reviews_df.date<window_end) & (reviews_df.date>window_start)].mean()
        d.append({'date':current_date,'polarity':polarity_average})
        window_start += time_delta
        window_end   += time_delta
        current_date += time_delta
    return pd.DataFrame(d)

def display_polarity_over_time(restaurant_list):
    """
    This function plots review sentiment polarity over time along with polarity running average 
    for all restaurants in restaurant_list.
    INPUT:
    restaurant_list = list of Restaurant Objects used to plot review sentiment polarity.
    """
    plt.figure(figsize=(20,8))
    for rest in restaurant_list:
        date_polarities = rest.get_review_polarities_by_date()
        sns.lineplot(x='date',y='polarity',data=date_polarities,alpha=0.3,label=rest.name)
        polarity_averages = get_polarity_over_time(date_polarities)
        sns.lineplot(x='date',y='polarity',data=polarity_averages,label=f"{rest.name} (average)")

def get_idf(doc_list, tokenized=False, ngram_range=(1,3)):
    """
    This function cleans, stops and tokenizes a list of documents (or takes in a pre-tokenized list) 
    and returns the IDF table.
    INPUTS:
    doc_list  = The list of texts to be analyzed.
    tokenized = Whether or not the documents are pre-cleaned and tokenized.
    
    OUTPUT:
    IDF table.
    """
    if tokenized:
        cleaned_tokenized = doc_list
    else:
        cleaned_tokenized = doc_list.map(clean_text).map(my_tokenizer).map(remove_stopwords)
        
    #TFIDF Vectorizer (settings for count vectorizer included)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                       tokenizer=dummy_function,
                                       preprocessor=dummy_function,
                                       token_pattern=None)
    #Fit all Docs
    tfidf_vectors=tfidf_vectorizer.fit_transform(cleaned_tokenized)
    #Return IDF dataframe
    scores = pd.DataFrame(tfidf_vectorizer.idf_, index=tfidf_vectorizer.get_feature_names(),columns=["idf_weight"])
    scores.sort_values(by=["idf_weight"], inplace=True)
    return scores


def get_tfidf(doc_list, tokenized=False, ngram_range=(1,3)):
    """
    This function cleans, stops and tokenizes a list of documents (or takes in a pre-tokenized list) 
    and returns the TF-IDF table.
    INPUTS:
    doc_list  = The list of texts to be analyzed.
    tokenized = Whether or not the documents are pre-cleaned and tokenized.
    
    OUTPUT:
    TF-IDF table.
    """
    if tokenized:
        cleaned_tokenized = doc_list
    else:
        cleaned_tokenized = doc_list.map(clean_text).map(my_tokenizer).map(remove_stopwords)
    
    #TFIDF Vectorizer (settings for count vectorizer included)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                       tokenizer=dummy_function,
                                       preprocessor=dummy_function,
                                       token_pattern=None)
    #Fit all Docs
    tfidf_vectors = tfidf_vectorizer.fit_transform(cleaned_tokenized)
    d = (dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_)))
    #d = (dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectors.toarray())))
    #Return TF/IDF dataframe
    scores = pd.DataFrame(d, index=['tfidf_weight']).T
    scores.sort_values(by='tfidf_weight', ascending=False, inplace=True)
    return scores

def get_tfidf_vectors(doc_list, tokenized=True, ngram_range=(1,1)):
    if not tokenized:
        doc_list = doc_list.map(clean_text).map(my_tokenizer).map(remove_stopwords)
        
    #TFIDF Vectorizer (settings for count vectorizer included)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                       tokenizer=dummy_fun,
                                       preprocessor=dummy_fun,
                                       token_pattern=None)
    #Fit all Docs
    return tfidf_vectorizer.fit_transform(doc_list)


def get_tfidf_scores(vectorizer, document):
    doc_vector = vectorizer.transform([document])
    df = pd.DataFrame(doc_vector.T.todense(), index=vectorizer.get_feature_names(), columns=["score"])
    return df.sort_values(by=["score"],ascending=False)

def dummy_function(doc):
    """
    Dummy function to be used in TfidfVectorizer so that I can use my own text cleaner and tokenizer.
    """
    return doc

def tsne_plot_words(model, n_words, positive, negative):
    #Get most similar words
    word_list = [w[0] for w in model.wv.most_similar(
        positive=positive, negative=negative, topn=n_words)] + positive + negative
    
    #Create TSNE model and plot it
    labels = []
    tokens = []

    for word in word_list:
        tokens.append(model[word])
        labels.append(word)
    
    #tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.show()