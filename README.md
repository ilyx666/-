# -
#  Алгоритмы кодирования категориальных признаков. В каком случае какой алгоритм применять
- LabelEncoder и OneHotEncoder работают только с категориальными функциями
- Сначала нам нужно извлечь категориальных персонажей, используя логическую маску.
```python
categorical_feature_mask = X.dtypes==object# filter categorical columns using mask and turn it into a list
categorical_cols = X.columns[categorical_feature_mask].tolist()
```
  # lABELeNCODER
  1.Создание экземпляра объекта LabelEncoder
  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  ```
  2.Примените LabelEncoder к каждому из категориальных столбцов
  ```python
  X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))X[categorical_cols].head(10)
  ```
  # OneHotEncoder
  1. Создание экземпляра объекта OneHotEncode
  ```python
  from sklearn.preprocessing import OneHotEncoder
  ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) 
  ```
  2.Применить OneHotEncoder к DataFrame
  ```python
  X_ohe = ohe.fit_transform(X)
  ```
# Алгоритм построения модели линейной регрессии
  1. Импортируем библиотеки
  2. Предоставляем данные
  3. Создаем модель
  4. Тренируем модель  и расчитаем коэфициенты детерменации для тренировочной и тестовой выборки
    - Коэффициент детерминации для модели с константой принимает значения от 0 до 1.
    - Чем ближе значение коэффициента к 1, тем сильнее зависимость. 
      1. коэффициент детерминации должен быть хотя бы не меньше 50 %.
      2. Модели с коэффициентом детерминации выше 80 % можно признать достаточно хорошими.
      3. Значение коэффициента детерминации 1 означает функциональную зависимость между переменными.
  5.Повышаем точность нашей модели
# Алгоритм построения моделей кластеризации
  # K-Means
  1.Импортируем библиотеки 
  2.Предоставляем данные
  3.Находим оптимальное количество кластеров(по методу локтя, где угол меняется больше всего)
  4.Тренируем модель
  5.Визуализируем кластеры
  # Hierarchical Clustering
  1.Импортируем библиотеки 
  2.Предоставляем данные
  3.Находим оптимальное количество кластеров(по дендрограмме, визуально смотрим)
  4.Тренируем модель
  5.Визуализируем модель
  
