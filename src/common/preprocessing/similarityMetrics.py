#   * Hellinger
#   * Kullback-Leibler
#   * Jaccard

from gensim.corpora import Dictionary
from gensim.models import ldamodel, LdaModel
from gensim.matutils import hellinger
from gensim.matutils import kullback_leibler
from gensim.matutils import jaccard
import numpy as np

from gensim.test.utils import datapath
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'\w+')
nltk.download('stopwords')
all_stopwords = stopwords.words('english')


class SimilarityMetrics():

    def __init__(self, file, texts):
        if os.path.isfile(os.getcwd() + file):
            temp_file = datapath(os.getcwd()+file)    
            self.model = LdaModel.load(temp_file)
        else:
            self.dictionary = Dictionary(texts)
            self.corpus = [self.dictionary.doc2bow(text) for text in texts]
            self.model = ldamodel.LdaModel(self.corpus, id2word=self.dictionary, num_topics=2, minimum_probability=1e-8)
            # Save model to disk.
            if not os.path.isdir(os.getcwd() +'/resource'):
                os.mkdir(os.getcwd() +'/resource')
            temp_file = datapath(os.getcwd() + "/resource/model_lda_preprocess")
            self.model.save(temp_file)
        

    def getJaccard(self,text1, text2):
        bow_text1 = self.model.id2word.doc2bow(text1)
        bow_text2 = self.model.id2word.doc2bow(text2)
        return jaccard(bow_text1, bow_text2)


    def getHellinger(self,text1, text2):
        bow_text1 = self.model.id2word.doc2bow(text1)
        bow_text2 = self.model.id2word.doc2bow(text2)
        lda_bow_text1 = self.model[bow_text1]
        lda_bow_text2 = self.model[bow_text2]
        return hellinger(lda_bow_text1, lda_bow_text2)

    def getKullback_leibler(self,text1, text2):
        bow_text1 = self.model.id2word.doc2bow(text1)
        bow_text2 = self.model.id2word.doc2bow(text2)
        lda_bow_text1 = self.model[bow_text1]
        lda_bow_text2 = self.model[bow_text2]
        return np.float64(kullback_leibler(lda_bow_text1, lda_bow_text2))

    def allMetrics(self, doc1, doc2):
        return self.getHellinger(doc1, doc2), self.getJaccard(doc1, doc2), self.getKullback_leibler(doc2, doc1)


    def combine_head_and_body(self, headlines, bodies, use_stopword):
        head_and_body = ''
        if use_stopword:
            head_and_body = [self.removeStopWord(self.getLemma(headline + " " + body)) for i, (headline, body) in
                             enumerate(zip(headlines, bodies))]            
        else:            
            head_and_body = [self.getLemma(headline + " " + body) for i, (headline, body) in
                             enumerate(zip(headlines, bodies))]            
        return head_and_body


    