import nltk
import spacy
from transformers import AutoTokenizer

nltk.download('punkt_tab')

class Tokenizer:
    def __init__(self, library):
        # Выбор библиотеки для токенизации
        self.library = library
        if library == 'nltk':
            # Использование NLTK для токенизации
            self.tokenizer = nltk.tokenize.WordPunctTokenizer()
        elif library == 'spacy':
            # Использование spacy для токенизации
            spacy.cli.download("en_core_web_sm")
            self.tokenizer = spacy.load('en_core_web_sm')
        elif library == 'transformers':
            # Использование transformers для токенизации
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            # Ошибка, если библиотека не поддерживается
            raise ValueError(f'Unsupported library: {library}')

    def tokenize(self, text):
        # Преобразование текста в токены
        # Проверка, что текст является строкой и не пустой
        if not isinstance(text, str):
            raise TypeError(f'Expected a string, got {type(text)}')
        if self.library == 'nltk':
            # Использование NLTK для токенизации
            tokens = self.tokenizer.tokenize(text)
        elif self.library == 'spacy':
            # Использование spacy для токенизации
            tokens = [token.text for token in self.tokenizer(text)]
        elif self.library == 'transformers':
            # Использование transformers для токенизации
            tokens = self.tokenizer.tokenize(text)
        return tokens

    def __str__(self):
        # Вывод информации об объекте класса
        return f'Tokenizer(library={self.library})'
