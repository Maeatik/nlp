import nltk
from nltk.stem import WordNetLemmatizer
import spacy
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class Lemmatizer:
    def __init__(self, library):
        # Выбор библиотеки для лемматизации
        self.library = library
        if library == 'nltk':
            # Использование NLTK для лемматизации
            self.lemmatizer = WordNetLemmatizer()
        elif library == 'spacy':
            # Использование spacy для лемматизации
            self.lemmatizer = spacy.load("en_core_web_sm")
        else:
            # Ошибка, если библиотека не поддерживается
            raise ValueError(f'Unsupported library: {library}')

    def lemmatize(self, text):
        # Лемматизация текста
        # Проверка, что текст является строкой и не пустой
        if not isinstance(text, str):
            raise TypeError(f'Expected a string, got {type(text)}')
        text = text.lower()
        text = re.sub('[!?@#$.,]', '', text)
        if self.library == 'nltk':
            # Использование NLTK для лемматизации
            tokens = nltk.word_tokenize(text)
            lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        elif self.library == 'spacy':
            # Использование spacy для лемматизации
            doc = self.lemmatizer(text)
            lemmas = [token.lemma_ for token in doc]
        return lemmas

    def __str__(self):
        # Вывод информации об объекте класса
        return f'Lemmatizer(library={self.library})'
