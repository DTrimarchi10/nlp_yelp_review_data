import pandas as pd
import re
import nltk
from gensim.summarization.textcleaner import split_sentences
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from textblob import TextBlob

#Define real words according to nltk corpus english vocabulary.
#This function is used for the remove_gibberish() method in the Review class
vocab_en = set(w.lower() for w in nltk.corpus.words.words())

#DEFINE CLEAN TEXT FUNCTION
def clean_text(text_block):
    """
    This function cleans a block of text.
    INPUT:
    text_block = The block of text to clean.
    OUTPUT:
    A block of text stripped of punctuation and made lowercase.
    """
    #This represents a non-breaking space in the text block that needs to be removed.
    text_block = text_block.replace(u'\xa0', u' ')
    #Replace & with and.
    text_block = text_block.replace('&',' and ')
    #Some reviews forget to have a space after a period. I want to force a space.
    text_block = text_block.replace('.',' ')
    #Replace underscores with spaces.
    text_block = text_block.replace('_',' ')
    #Remove all other punctuation and numbers (except apostrophe).
    text_block = re.sub(r'[^\w\s\']|\d',' ',text_block.lower())
    #Remove multiple spaces and make a single space. Also remove leading/trailing spaces.
    text_block = re.sub(' +', ' ', text_block.strip())
    return text_block

#DEFINE REVIEW CLASS
class Review:
    """
    Review class is used to contain the date, text, star_rating, and sentiment polarity of a review.
    """

    def __init__(self, text, date, star_rating):
        self.text = split_sentences(text)
        self.date = date
        self.star_rating = star_rating
        self.tokenized = False
        self.polarity = 0
    
    def get_pretty_review(self):
        """
        This method will return the review text in a sentence format with the beginning of each sentence capitalized.
        """
        if self.tokenized:
            return '. '.join([' '.join(sentence).capitalize() for sentence in self.text])
        else:
            return '. '.join([sentence.capitalize() for sentence in self.text])
        
    def get_review(self):
        """
        This method will return the entire review as a single string.
        """
        if self.tokenized:
            return ' '.join([' '.join(sentence) for sentence in self.text])
        else:
            return ' '.join(self.text)
        
    def get_all_tokens(self):
        """
        This method will return all tokens for the review. Sentence structure is not preserved.
        """
        if self.tokenized:
            return [token for sentence in self.text for token in sentence]
        else:
            return None
    
    def clean_text(self):
        """
        This method applies text cleaning to the review text. It uses the clean_text function.
        It should be called before tokenizing but can be called at any time.
        """
        if self.tokenized:
            self.text = [[clean_text(token) for token in sentence] for sentence in self.text]
        else:
            self.text = [clean_text(sentence) for sentence in self.text]
        
    def remove_stopwords(self, stopwords):
        """
        This method removes stopwords from the review text.
        INPUT: stopwords = List of stopwords to be removed.
        """
        if self.tokenized:
            self.text = [[token for token in sentence if token not in stopwords] for sentence in self.text]
        else:
            self.text = [' '.join([word for word in sentence.split(sep=" ") 
                                   if word not in stopwords]) for sentence in self.text]
            
    def remove_gibberish(self, vocab=vocab_en):
        """
        This method removes words from review text that are not in the recognized vocabulary.
        INPUT: vocab = The list of words in the vocabulary. Default is nltk corpus english.
        """
        if self.tokenized:
            self.text = [[token for token in sentence if token in vocab] for sentence in self.text]
        else:
            self.text = [' '.join([word for word in sentence.split(sep=" ") 
                                   if word in vocab]) for sentence in self.text]

    def drop_short_sentences(self, length):
        """
        This method removes short sentences.
        INPUT: length = All sentences equal to or shorter than length will be dropped.
        If tokenized, length refers to the number of words in the sentence.
        If not tokenized, length refers to the number of characters in the sentence.
        """
        #If tokenized, the length refers to number of words in the sentence.
        #If not tokenized, the length refers to number of characters in the sentence.
        self.text = [sentence for sentence in self.text if len(sentence)>length]
    
    def drop_short_words(self, length):
        """
        This method removes short words.
        INPUT: length = All words equal to or shorter than length will be dropped.
        """
        if self.tokenized:
            self.text = [[word for word in sentence if len(word)>length]for sentence in self.text]
        else:
            self.text = [' '.join([word for word in sentence.split(sep=" ") 
                                   if len(word)>length]) for sentence in self.text]
    
    def apply_phraser(self, phraser):
        """
        Apply pre-fit phraser to the review text.
        INPUT: phraser = Phraser to be applied.
        """
        self.text = [phraser[sentence] for sentence in self.text]
        
    def tokenize(self, tokenizer):
        """
        Apply a tokenizer to the review text.
        INPUT: tokenizer = Tokenizer to be applied.
        """
        self.text = [tokenizer(sentence) for sentence in self.text]
        self.tokenized = True
        
    def assign_polarity(self):
        """
        Assigns polarity to the review text using TextBlob.sentiment.polarity.
        """
        self.polarity = TextBlob(self.get_review()).sentiment.polarity
        
    def to_dict(self):
        """
        Returns a dictionary containing all review information.
        Keys: date, star_rating, polarity, review_text.
        """
        return {'date':self.date,
                'star_rating':self.star_rating,
                'polarity':self.polarity,
                'review_text':self.text
               }

#DEFINE RESTAURANT CLASS
class Restaurant:

    def __init__(self, name, business_index ,address, categories, price_range, star_rating):
        self.name = name
        self.biz_id = business_index
        self.address = address
        self.categories = categories
        self.price_range = price_range
        self.star_rating = star_rating
        self.reviews = []
        self.keywords = {1.0:[],2.0:[],3.0:[],4.0:[],5.0:[]}
        
    def add_review(self, review_object):
        """
        Add a single review object to the restaurant's reviews list.
        INPUT: review_object = review to be added.
        """
        self.reviews.append(review_object)
    
    def set_reviews(self, review_list):
        """
        Set all reviews for a restaurant.
        This function overwrites any reviews previously stored for the restaurant.
        INPUT: review_list = list of Review objects to be applied.
        """
        self.reviews = review_list
        
    def get_document(self, star_rating=0):
        """
        Returns a document containing all tokens for all reviews.
        INPUT: star_rating = If zero, return document for all reviews, 
        otherwise return document for specific star_rating level (1-5).
        """
        all_reviews = []
        if star_rating==0:
            #return doc of all reviews
            for review in self.reviews:
                all_reviews.extend(review.get_all_tokens())
        else:
            #return doc of only reviews matching star rating
            for review in self.reviews:
                if review.star_rating==star_rating:
                    all_reviews.extend(review.get_all_tokens())
        return all_reviews
    
    def get_review_sentences(self, star_rating=0):
        """
        Returns a list containing all sentences for all reviews.
        INPUT: star_rating = If zero, return document for all reviews, 
        otherwise return sentences for specific star_rating level (1-5).
        """
        review_sentences = []
        if star_rating == 0:
            for review in self.reviews:
                review_sentences.extend(review.text)
        else:
            for review in self.reviews:
                if review.star_rating==star_rating:
                    review_sentences.extend(review.text)
        return review_sentences
    
    def get_review_objects(self):
        """
        Return list of review objects.
        """
        return self.reviews
    
    def get_keywords(self):
        """
        Return restaurant keywords.
        OUTPUT:
        List of dictionaries containing the following keys: Restaurant Id, star_rating, word, score.
        """
        keywords_list = []
        for rating in self.keywords.keys():
            for word in self.keywords[rating]:
                word['star_rating'] = rating
                word['Restaurant Id'] = self.biz_id
                keywords_list.append(word)
        return keywords_list
    
    def is_in_category(self, category_name):
        """
        Return True if restaurant is part of the category.
        INPUT: category_name = Text string to check for.
        """
        if category_name in self.categories:
            return True
        else:
            return False
        
    def get_review_polarities_by_date(self):
        """
        Returns a Pandas DataFrame containing the date and polarity for each review.
        Columns = date, polarity
        """
        #return dates[], polarities[]
        #return tuple(map(list,zip(*[(review.date, review.polarity) for review in self.reviews])))
        return pd.DataFrame.from_records([(review.date, review.polarity) for review in self.reviews], 
                                       columns =['date', 'polarity'])
            
    def set_keywords(self, star_rating, words, values): 
        """
        Set keywords variable for a restaurant.
        INPUTS: 
        star_rating = The star rating that the keywords apply to.
        words = List of words to be added to the keywords list.
        values = List of values (scores) for each word.
        """
        self.keywords[star_rating] = [{'word':word, 'score':value} for word, value in zip(words, values)]
        
    def to_dict(self):
        """
        Returns a dictionary containing restaurant information without the reviews.
        Keys: Business Name, Business Address, Business Index, Category, Price Range, Star Rating.
        """
        return {'Business Name':self.name,
                'Business Address':self.address,
                'Business Index':self.biz_id,
                'Category':self.categories,
                'Price Range':self.price_range,
                'Star Rating':self.star_rating
               }
