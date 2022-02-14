from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk
nltk.download('punkt')

from sumy.summarizers.text_rank import TextRankSummarizer as TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


class TextRank_Summarizer:
    def __init__(self, language, sentences_count):
        self.summary = None
        self.language = language
        self.stemmer = Stemmer(self.language)
        self.textRankSummarizer = TextRankSummarizer(self.stemmer)
        self.textRankSummarizer.stop_words = get_stop_words(self.language)
        self.sentences_count = sentences_count
        


    def cal_summary(self, body):          
        parser = PlaintextParser.from_string(body, Tokenizer(self.language))    
        summary = self.textRankSummarizer(parser.document, self.sentences_count)
        
        if summary != None:
            summary = self.toString(summary)
    
        return summary
    
    def toString(self, tupla):
        data = ''
        for row in tupla:
            data += row._text +" "
        return data