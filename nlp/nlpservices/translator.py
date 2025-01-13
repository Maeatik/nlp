from transformers import (
    MarianMTModel,
    MarianTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)


class Translator:
    def __init__(self, model_name: str, source_lang: str = None, target_lang: str = None):
        self.model_name = model_name.lower()
        
        if self.model_name == "marian":
            if not source_lang or not target_lang:
                raise ValueError("Для MarianMT необходимо указать source_lang и target_lang.")
            marian_model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            self.tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
            self.model = MarianMTModel.from_pretrained(marian_model_name)
        
        elif self.model_name == "t5":
            t5_model_name = "t5-small"  # Можно заменить на t5-base или t5-large
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        
        elif self.model_name == "bart":
            bart_model_name = "facebook/bart-large-cnn"  # Используем BART
            self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)
            self.model = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        else:
            raise ValueError("Поддерживаются только модели 'marian', 't5', 'bart'.")
    
    def translate(self, text: str) -> str:
        if self.model_name == "marian":
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        elif self.model_name == "t5":
            input_text = f"translate English to Russian: {text}"
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        elif self.model_name == "bart":
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
