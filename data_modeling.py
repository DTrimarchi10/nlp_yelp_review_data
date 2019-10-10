import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from keras.preprocessing import text, sequence


def load_GloVe_vectors(file, vocab):
    """
    This function will load Global Vectors for words in vocab from the specified GloVe file.
    
    INPUT:
    file  = The path/filename of the file containing GloVe information.
    vocab = The list of words that will be loaded from the GloVe file.
    """
    glove = {}
    with open(file, 'rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0].decode('utf-8')
            if word in vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                glove[word] = vector
    return glove


#Creating Mean Word Embeddings using Mean Embedding Vectorizer class
class W2vVectorizer(object):
    """
    This class is used to provide mean word vectors for review documents. 
    This is done in the transform function which is used to generate mean vectors in model pipelines.
    The class has both fit and transform functions so that it may be used in an sklearn Pipeline.
    """
    
    def __init__(self, model):
        self.w2v = model
        
        #If using GloVe the model is in a dictionary format
        if isinstance(model, dict):
            if len(model) == 0:
                self.dimensions = 0
            else:
                self.dimensions = len(model[next(iter(model))])
        #Otherwise, using gensim keyed vector
        else:
            self.dimensions = model.vector_size
    
    # Need to implement a fit method as required for sklearn Pipeline.
    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        This function generates a w2v vector for a set of tokens. This is done by taking 
        the mean of each token in the review.
        """
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                    or [np.zeros(self.dimensions)], axis=0) for words in X])
    
class KerasTokenizer(object):
    """
    This class is used to fit text and convert text to sequences for use in a Keras NN Model.
    The class has both fit and transform functions so that it may be used in an sklearn Pipeline.
    num_words = max number of words to keep.
    maxlen  = max length of all sequences.
    """
    def __init__(self, num_words=20000, maxlen=100):
        self.tokenizer = text.Tokenizer(num_words=num_words)
        self.maxlen = maxlen
        
    def fit(self, X, y):
        self.tokenizer.fit_on_texts(X)
        return self
        
    def transform(self, X):
        return sequence.pad_sequences(self.tokenizer.texts_to_sequences(X), maxlen=self.maxlen)
    
class KerasModel(object):
    """
    This class is used to fit and transform a keras model for use in an sklearn Pipeline.
    """
    def __init__(self, model, epochs=3, batch_size=32, validation_split=0.1):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
    def set_params(self, epochs=3, batch_size=32, validation_split=0.1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        return self
    
    def fit(self, X, y):
        y_dummies = pd.get_dummies(y).values
        self.labels = np.array(pd.get_dummies(y).columns)
        self.model.fit(X, y_dummies, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)
        return self
    
    def transform(self, X):
        return X
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return [self.labels[idx] for idx in y_pred.argmax(axis=1)]
    
    def summary(self):
        self.model.summary()
        
    
def custom_accuracy(y_true, y_pred, threshold=1, credit_given=0.5):
    """
    If y_pred is off by threshold or less, give partial credit according to credit_given.
    INPUTS:
    y_true       = True label.
    y_pred       = Predicted label.
    threshold    = Threshold for giving credit to inaccurate predictions. (=1 gives credit to predictions off by 1)
    credit_given = Partial credit amount for close but inaccurate predictions. (=0.5 give 50% credit)
    """
    predicted_correct = sum((y_true-y_pred)==0)
    predicted_off = sum(abs(y_true-y_pred)<=threshold) - predicted_correct
    custom_accuracy = (predicted_correct + credit_given*predicted_off)/len(y_true)
    return custom_accuracy


def get_gridsearch_result(name, estimator, param_grid, X_train, X_test, y_train, y_test, cv=4, scoring='accuracy'):
    """
    This function fits a GridSearchCV model and populates a dictionary containing the best model and accuracy results.
    
    INPUTS:
    name       = The name of the gridsearch model desired. It will be used as a key in the returned dictionary object.
    estimator  = The model which will be passed into GridSearchCV. It can be a pipeline model or a base model.
    param_grid = The parameter grid to be used by GridSearchCV.
    X_train, X_test, y_train, y_test = train test split of data(X) and target(y).
    cv         = Number of cross validations to perform.
    
    RETURN:
    Dictionary containing the fitted GridSearchCV model as well as summary metrics.
    Dictionary keys are: name, model, model params, accuracy train, accuracy test, 
                         custom accuracy train, custom accuracy test.
    """
    grid_clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_clf.fit(X_train,y_train)

    result = {}
    
    result['name'] = name
    result['model'] = grid_clf.best_estimator_
    result['model params'] = grid_clf.best_params_

    y_pred_train = grid_clf.predict(X_train)
    y_pred_test  = grid_clf.predict(X_test)

    result['accuracy train']  = round(accuracy_score(y_train,y_pred_train),4)
    result['accuracy test']   = round(accuracy_score(y_test,y_pred_test),4)
    result['custom accuracy train'] = round(custom_accuracy(y_train,y_pred_train),4)
    result['custom accuracy test']  = round(custom_accuracy(y_test,y_pred_test),4)

    return result


def gridsearch_all_models(models, X_train, X_test, y_train, y_test, cv=4, scoring='accuracy'):
    """
    This function will perform a grisearch on a list of models, output the time taken, 
    and return a list of results dictionaries for each gridsearch model.
    
    INPUTS:
    models = List of tuples in the form (name, model, param_grid)
        name  = text name of the model
        model = model of pipeline model
        param_grid = parameters to be used for gridsearch
    X_train, X_test, y_train, y_test = train test split of data(X) and target(y).
    
    RETURNS:
    List of dictionaries containing gridsearch model, selected parameters, and accuracy scores.
    """
    
    print("GRIDSEARCH AND SCORE ALL MODELS:")
    start = time.time()
    results = []

    for name, model, param_grid in models:
        start_model = time.time()
        print("  ", name, end='')
        results.append(get_gridsearch_result(name=name,
                                             estimator=model,
                                             param_grid=param_grid,
                                             X_train=X_train,
                                             X_test=X_test, 
                                             y_train=y_train,
                                             y_test=y_test,
                                             cv=cv,
                                             scoring=scoring))

        end_model = time.time()
        print(":\t time", time.strftime('%H:%M:%S', time.gmtime(end_model-start_model)))

    end = time.time()
    print("TOTAL TIME:", time.strftime('%H:%M:%S', time.gmtime(end-start)))
    return results