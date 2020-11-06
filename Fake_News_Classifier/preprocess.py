import en_core_web_sm
nlp = en_core_web_sm.load()
import gensim
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer

class Preprocesser:
    def get_text_lst(self,dataframe):
        """
        Function to convert text to list and remove emails, new lines and single quotes
        """
        text_lst = dataframe.text.values.tolist()
        text_lst = [re.sub(r'\S*@\S*\s?', '', sentence) for sentence in text_lst]
        text_lst = [re.sub(r'\s+', ' ', sentence) for sentence in text_lst]
        text_lst = [re.sub(r"\'", "", sentence) for sentence in text_lst]
        return text_lst

    def sentence_to_words(self,sentences):
        """
        Function to convert sentence to words and remove punctuations
        """
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    
    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """
        Function to lemmatize words using allowed postages, specific to the usage
        """
        texts_res = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_res.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_res
    
    def vectorizer(self,dataframe,text_lemma):
        """
        Function to vectorize news articles using lemmatized text
        """
        vectorizer = CountVectorizer(analyzer='word',       
                             min_df=5,                      # minimum reqd occurences of a word 
                             stop_words='english',           # remove stop words
                             lowercase=True,                 # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}')# num chars > 3
        vec = vectorizer.fit(dataframe['text'])
        text_vectorized = vec.fit_transform(text_lemma)
        return text_vectorized,vec

    def get_top_n_words(self,text_vectorized,vec,n):
        """
        Function to get top n words for news articles
        """
        sum_words = text_vectorized.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        top_n_words = words_freq[0:n]
        final_top_n = [item[0] for item in top_n_words]
        return final_top_n
    
