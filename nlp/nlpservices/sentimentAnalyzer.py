from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

class SentimentAnalysis:
    def __init__(self, library):
        # Выбор библиотеки для стемминга
        self.library = library
        if library=='transformers':
            self.model_name = "cointegrated/rubert-tiny2-cedr-emotion-detection"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.classifier = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer)
        else:
            # Ошибка, если библиотека не поддерживается
            raise ValueError(f'Unsupported library: {library}')
        
    def analyze_sentiment(self, text):
        # Проверка, что текст является строкой и не пустой
        if not isinstance(text, str):
            raise TypeError(f'Expected a string, got {type(text)}')
        elif self.library == 'transformers':
            result = self.classifier(text, return_all_scores=True)
            return result   
        
    def __str__(self):
        # Вывод информации об объекте класса
        return f'SentimentAnalysis(library={self.library})'
