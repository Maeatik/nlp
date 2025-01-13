from django import forms

class CalculatorForm(forms.Form):
    input_text = forms.CharField(label='Исходный текст', widget=forms.Textarea)    
    operation = forms.ChoiceField(
        choices=(   
            # Список операций с ключами на английском и значениями на русском
            ("Lemmatization", "Нормализация"),
            ("Tokenization", "Токенизация"),
            ("Word2Vec", "Векторизация"),
            ("Classification", "Классификация"),
            ("Clusterisation", "Кластеризация"),
            ("MachineTranslation", "Машинный перевод")
        ),
        widget=forms.Select(attrs={'onchange': 'this.form.submit();'}),  # Добавляем атрибут onchange
        label='Выберите операцию'
    )
    librarychoice = forms.ChoiceField(
        choices=(),
        widget=forms.Select(),  # Используем выпадающий список для выбора библиотеки
        label='Выберите метод'
    )
    output_text = forms.CharField(label='Результат', widget=forms.Textarea, required=False, disabled=True)

    def __init__(self, libraries, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Получаем значение поля operation из данных формы
        operation = self.data.get("operation")
        # Если значение не пустое, устанавливаем соответствующие библиотеки для поля librarychoice
        if operation:
            self.fields["librarychoice"].choices = [(lib, lib) for lib in libraries[operation]]
