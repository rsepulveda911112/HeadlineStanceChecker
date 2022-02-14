import re
import spacy
from multiprocessing import cpu_count, Pool

nlp = spacy.load('en_core_web_lg')
all_stopwords = nlp.Defaults.stop_words

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def mergeSentences(sentences):
    text = ''
    for row in sentences:
        text += row +" "
    text = text[:-1]
    return text

def getLemma(text):
    doc = nlp(text)
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_)
    return lemmas

def removeStopWord(tokens):
   return [word for word in tokens if not word in all_stopwords]


def combine(row):
    headline = row[2]
    body = row[3]
    stop_word = True
    headline_lema = getLemma(clean(headline))
    body_lema = getLemma(clean(body))
    if stop_word:
        headline_lema = removeStopWord(headline_lema)
        body_lema = removeStopWord(body_lema)
    combine = headline_lema + body_lema
    return combine, headline_lema, body_lema