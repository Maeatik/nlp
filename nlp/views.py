from django.shortcuts import render
from nlp.forms.forms import CalculatorForm
from nlp.nlpservices.middleware import Middleware

def calculate(request):
    middleware = Middleware()
    # Создаем словарь с библиотеками для каждой операции
    libraries = {
        "Lemmatization": ['nltk', 'spacy'],
        "Tokenization": ['nltk', 'spacy', 'transformers'],
        "Word2Vec": ['word2vec', 'spacy', 'tfidf_vectorizer', 'count_vectorizer'],
        "Classification": ['XGboost', 'CNN', 'RNN', 'transformers'],
        "Clusterisation": ['k-means', 'hierarchical'],
        "MachineTranslation": ['marian', 't5', 'bart']
    }
    if request.method == 'POST':
        form = CalculatorForm(libraries, request.POST)
        if form.is_valid():
            input_text = form.cleaned_data['input_text']
            operation = form.cleaned_data['operation']             
            librarychoice = form.cleaned_data['librarychoice']                 
            library = librarychoice.lower()
            if operation == 'Lemmatization':
                output_text = middleware.lematization(input_text, library)
            elif operation == 'Tokenization':
                output_text = middleware.tokenization(input_text, library)
            elif operation == 'Word2Vec':
                output_text = middleware.word2vec(input_text, library)
            elif operation == 'Classification':
                output_text = middleware.analyze_sentiment(input_text, library) 
            elif operation == 'Clusterisation':
                output_text = middleware.clusterization(input_text, library)    
            elif operation == 'MachineTranslation':
                output_text = middleware.translation(input_text, library)    
            else:
                output_text = 'Нет обработчика для задачи'                        
            form = CalculatorForm(libraries, initial={'input_text': input_text, 'operation': operation, 'librarychoice': librarychoice, 'output_text': output_text})
    else:
        form = CalculatorForm(libraries)
    return render(request, 'nlptask.html', {'form': form})
