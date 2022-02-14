from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from common.preprocessing.util import clean, mergeSentences, removeStopWord, getLemma


def combine_head_and_body(headline, body):
    head_and_body = headline + " " + body
    return [head_and_body]

def get_features(vocab, headline, body):
    vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                      norm="l2", stop_words='english')
    X_head = vectorizer_head.fit_transform([headline])

    vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                      norm="l2", stop_words='english')
    X_body = vectorizer_body.fit_transform([body])

    X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

    return X, X_head, X_body

def word_feature(headline, body):
    """
    Simple bag of words feature extraction with term freq of words as feature vectors, length 5000 head + 5000 body,
    concatenation of head and body, l2 norm and bleeding (BoW = train+test+holdout+unlabeled test set).
    """

    # create the vocab out of the BoW
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=True,
                            norm='l2',strip_accents='unicode')

    tfidf.fit_transform(combine_head_and_body(headline, body))
    vocab = tfidf.vocabulary_

    X, X_head, X_body = get_features(vocab, headline, body)
    cosine_similarity_value = cosine_similarity(X_head, X_body)[0][0]

    return X, cosine_similarity_value