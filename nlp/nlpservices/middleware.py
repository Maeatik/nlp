from nlp.nlpservices import sentimentAnalyzer
from nlp.nlpservices.clusterizer import Clusterizer
from nlp.nlpservices.lemmatizer import Lemmatizer
from nlp.nlpservices.tokenizer import Tokenizer
from nlp.nlpservices.translator import Translator
from nlp.nlpservices.wordEmbedding import WordEmbedding

class Middleware:
    @staticmethod
    def tokenization(text, library):
        tokenizer = Tokenizer(library)
        tokens = tokenizer.tokenize(text)
        return ' '.join(tokens)

    @staticmethod
    def lematization(text, library):
        lemmatizer = Lemmatizer(library)
        lemmas = lemmatizer.lemmatize(text)
        return ' '.join(lemmas)
    
    @staticmethod
    def word2vec(text, library):
        vectors = WordEmbedding(library)
        vects = vectors.vectorize(text)
        # Преобразование векторов в строки для объединения
        vects_str = [', '.join(map(str, vector)) for vector in vects]
        return vects_str
    
    @staticmethod
    def analyze_sentiment(text, library):        
        sentiment_analyzer = sentimentAnalyzer.SentimentAnalysis(library)
        result = sentiment_analyzer.analyze_sentiment(text)
        formatted_result = "\n".join([f"{item['label']}: {item['score']}" for item in result[0]])
        return formatted_result    
    
    @staticmethod
    def clusterization(text, library):
        clusterizer = Clusterizer(library)
        clusters = clusterizer.clusterize(text)
        return clusters
    
    @staticmethod
    def translation(text, library):
        translator = Translator(model_name=library, source_lang="en", target_lang="ru")
        return translator.translate(text)