from sklearn.neighbors import KNeighborsClassifier

# Обучающие данные (масса, текстура)
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# Соответствующие метки классов (яблоко - 0, апельсин - 1)
labels = [0, 0, 1, 1]

# Создаем модель k-NN с числом соседей равным 1
k = 1
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Обучаем модель
knn_classifier.fit(features, labels)

# Предсказываем класс для нового фрукта (масса 160 г, текстура шершавая)
new_fruit = [[160, 0]]
prediction = knn_classifier.predict(new_fruit)

if prediction[0] == 0:
    print("Фрукт - яблоко")
else:
    print("Фрукт - апельсин")
