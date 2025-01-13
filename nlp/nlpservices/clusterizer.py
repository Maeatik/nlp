from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import re

class Clusterizer:
    def __init__(self, library):
        # Выбор библиотеки для кластеризации
        self.library = library
        if library == 'k-means':
            # Использование к средних для кластеризации
            self.clusterizer = KMeans(n_clusters=3)
        elif library == 'hierarchical':
            # Использование иерархической кластеризации
            self.clusterizer = AgglomerativeClustering(n_clusters=3)
        else:
            # Ошибка, если библиотека не поддерживается
            raise ValueError(f'Unsupported library: {library}')

    def clusterize(self, text):
        # Кластеризация текста
        # Проверка, что текст является строкой и не пустой
        if not isinstance(text, str):
            raise TypeError(f'Expected a string, got {type(text)}')
        matches = re.findall(r"'(.*?)'", text)
        numeric_data = [list(map(float, match.split(', '))) for match in matches]
        vectors = np.array(numeric_data)
        if self.library == 'k-means':
            self.clusterizer.fit(vectors)
            return ", ".join(map(str, self.clusterizer.labels_))
        elif self.library == 'hierarchical':
            return ", ".join(map(str, self.clusterizer.fit_predict(vectors)))

    def __str__(self):
        # Вывод информации об объекте класса
        return f'Clusterizer(library={self.library})'
