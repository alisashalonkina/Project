# 4.	ПЕРВИЧНЫЙ АНАЛИЗ НАБОРА ТЕКСТОВЫХ ДАННЫХ

!pip install pymystem3 pandas matplotlib wordcloud nltk scikit-learn -q

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
import warnings
warnings.filterwarnings('ignore')

# Загрузка стоп-слов NLTK
nltk.download('stopwords', quiet=True)

# Инициализация Mystem
mystem = Mystem()

#  ЗАГРУЗКА ДАННЫХ

df = pd.read_csv('banki_reviews.csv', sep=';', encoding='utf-8')

print(f"\nДатасет загружен. Размер: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print("\nПервые 5 строк:")
print(df.head())

#  СОЗДАНИЕ ЦЕЛЕВОГО ПРИЗНАКА

def create_sentiment_label(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df_analysis = pd.DataFrame()
df_analysis['text'] = df['Review'].astype(str)
df_analysis['rating'] = df['Grade']
df_analysis['label'] = df_analysis['rating'].apply(create_sentiment_label)

df_analysis = df_analysis[df_analysis['text'].str.len() > 10]
df_analysis = df_analysis.reset_index(drop=True)

print(f"\nПодготовлено {len(df_analysis)} отзывов для анализа")
print(f"\nРаспределение по классам:")
print(df_analysis['label'].value_counts().sort_index())
print("0 - Negative, 1 - Neutral, 2 - Positive")

#  ОЧИСТКА ТЕКСТА

def clean_text(text):
    text = text.lower()                    # Приводим к нижнему регистру
    text = re.sub(r'[^а-яё\s]', '', text)  # Удаляем всё, кроме русских букв и пробелов
    text = re.sub(r'\s+', ' ', text)       # Заменяем множественные пробелы на один
    text = text.strip()                    # Удаляем пробелы в начале и конце
    return text


df_analysis['clean_text'] = df_analysis['text'].apply(clean_text)


print("\n" + "="*50)
print("ПРИМЕРЫ ОЧИСТКИ ТЕКСТА")
print("="*50)

for i in range(3):
    print(f"\nПример {i+1}:")
    print(f"ДО:   {df_analysis['text'].iloc[i][:150]}...")
    print(f"ПОСЛЕ: {df_analysis['clean_text'].iloc[i][:150]}...")

#  ЛЕММАТИЗАЦИЯ

def lemmatize_text(text):
    lemmas = mystem.lemmatize(text)
    result = ' '.join([w for w in lemmas if w.strip()])
    result = re.sub(r'\s+', ' ', result).strip()
    return result


neg_sample = df_analysis[df_analysis['label'] == 0].head(1000)
neu_sample = df_analysis[df_analysis['label'] == 1]  # все нейтральные (281)
pos_sample = df_analysis[df_analysis['label'] == 2].head(1000)

df_sample = pd.concat([neg_sample, neu_sample, pos_sample])
df_sample = df_sample.reset_index(drop=True)

print("\n" + "="*50)
print("ФОРМИРОВАНИЕ ВЫБОРКИ")
print("="*50)
print(f"Размер выборки для анализа: {len(df_sample)} отзывов")
print(f"Из них:")
print(f"  Negative: {len(df_sample[df_sample['label'] == 0])}")
print(f"  Neutral:  {len(df_sample[df_sample['label'] == 1])}")
print(f"  Positive: {len(df_sample[df_sample['label'] == 2])}")

print("\nЛемматизация отзывов...")
df_sample['lemmas'] = df_sample['clean_text'].apply(lemmatize_text)

print("ПРИМЕРЫ ЛЕММАТИЗАЦИИ")
for i in range(min(3, len(df_sample))):
    print(f"\nПример {i+1}:")
    print(f"ДО (очищенный):   {df_sample['clean_text'].iloc[i][:150]}...")
    print(f"ПОСЛЕ (леммы):     {df_sample['lemmas'].iloc[i][:150]}...")

#  ПОДСЧЕТ ЧАСТОТЫ СЛОВ (ДО СТОП-СЛОВ)


all_words = ' '.join(df_sample['lemmas']).split()
word_counts = Counter(all_words)
top_words = word_counts.most_common(20)

print("\n" + "="*50)
print("ТОП-10 САМЫХ ЧАСТЫХ СЛОВ (ДО УДАЛЕНИЯ СТОП-СЛОВ)")
print("="*50)
for word, count in top_words[:10]:
    print(f"{word}: {count}")

words, counts = zip(*top_words[:10])

plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='steelblue')
plt.title('Топ-10 самых частых слов в отзывах (до удаления стоп-слов)', fontsize=14)
plt.xlabel('Слова', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

all_text = ' '.join(df_sample['lemmas'])
wc = WordCloud(width=800, height=400, max_words=50, background_color='white',
               colormap='viridis', random_state=42).generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Облако слов для всех отзывов (до удаления стоп-слов)', fontsize=14)
plt.tight_layout()
plt.show()

#  УДАЛЕНИЕ СТОП-СЛОВ

russian_stopwords = set(stopwords.words('russian'))

# Добавляем дополнительные стоп-слова
additional_stopwords = {'банк', 'банка', 'банку', 'банке', 'озон', 'ozon',
                        'очень', 'такой', 'этот', 'все', 'даже', 'который',
                        'быть', 'сказать', 'стать', 'мочь', 'весь', 'еще'}
russian_stopwords.update(additional_stopwords)

print("\n" + "="*50)
print(f"Всего стоп-слов: {len(russian_stopwords)}")
print("="*50)

def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w not in russian_stopwords])

df_sample['no_stopwords'] = df_sample['lemmas'].apply(remove_stopwords)

print("\nПРИМЕРЫ ТЕКСТОВ ДО И ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ")
for i in range(min(3, len(df_sample))):
    print(f"\nПример {i+1}:")
    print(f"ДО (с леммами):   {df_sample['lemmas'].iloc[i][:150]}...")
    print(f"ПОСЛЕ (без стоп):  {df_sample['no_stopwords'].iloc[i][:150]}...")

#  ПОДСЧЕТ ЧАСТОТЫ ПОСЛЕ СТОП-СЛОВ

all_words_clean = ' '.join(df_sample['no_stopwords']).split()
word_counts_clean = Counter(all_words_clean)
top_words_clean = word_counts_clean.most_common(20)

print("\n" + "="*50)
print("ТОП-10 САМЫХ ЧАСТЫХ СЛОВ (ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ)")
print("="*50)
for word, count in top_words_clean[:10]:
    print(f"{word}: {count}")

words_clean, counts_clean = zip(*top_words_clean[:10])

plt.figure(figsize=(12, 6))
plt.bar(words_clean, counts_clean, color='darkorange')
plt.title('Топ-10 самых частых слов в отзывах (ПОСЛЕ удаления стоп-слов)', fontsize=14)
plt.xlabel('Слова', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#  TF-IDF ВЕКТОРИЗАЦИЯ

vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df_sample['no_stopwords'])
feature_names = vectorizer.get_feature_names_out()

print("\n" + "="*50)
print("TF-IDF ВЕКТОРИЗАЦИЯ")
print("="*50)
print(f"Размер словаря: {len(feature_names)} слов")
print(f"Размер матрицы TF-IDF: {tfidf_matrix.shape}")

#  ИНФОРМАЦИОННЫЙ ПОИСК

def search_texts(query, df_sample, vectorizer, tfidf_matrix, top_n=5):
    query_clean = clean_text(query)
    query_lemmas = lemmatize_text(query_clean)
    query_no_stop = remove_stopwords(query_lemmas)
    query_vec = vectorizer.transform([query_no_stop])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'индекс': idx,
            'текст_оригинал': df_sample['text'].iloc[idx][:250],
            'рейтинг': df_sample['rating'].iloc[idx],
            'класс': df_sample['label'].iloc[idx],
            'похожесть': similarities[idx]
        })
    return results

queries = ["карта заблокирована", "спасибо сотрудникам", "проблема с чатом поддержки", "навязали страховку"]

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ИНФОРМАЦИОННОГО ПОИСКА")
print("="*50)

for query in queries:
    print(f"\n\nЗапрос: \"{query}\"")
    print("-" * 40)
    results = search_texts(query, df_sample, vectorizer, tfidf_matrix, top_n=3)
    
    for i, res in enumerate(results, 1):
        if res['класс'] == 0:
            sentiment = "НЕГАТИВ"
        elif res['класс'] == 1:
            sentiment = "НЕЙТРАЛЬНО"
        else:
            sentiment = "ПОЗИТИВ"
        
        print(f"\nРезультат {i}:")
        print(f"  Похожесть: {res['похожесть']:.4f}")
        print(f"  Рейтинг: {res['рейтинг']} → {sentiment}")
        print(f"  Текст: {res['текст_оригинал']}...")

#  АНАЛИЗ ПО КЛАССАМ ТОНАЛЬНОСТИ

print("\n" + "="*50)
print("ТОП-10 СЛОВ ПО КЛАССАМ ТОНАЛЬНОСТИ")
print("="*50)

classes = {0: "Отрицательные", 1: "Нейтральные", 2: "Положительные"}

for label, name in classes.items():
    class_texts = df_sample[df_sample['label'] == label]['no_stopwords']
    if len(class_texts) > 0:
        all_words_class = ' '.join(class_texts).split()
        word_counts_class = Counter(all_words_class)
        top_class_words = word_counts_class.most_common(10)
        print(f"\n{name} (всего {len(class_texts)} отзывов):")
        for word, count in top_class_words:
            print(f"  {word}: {count}")

#  ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ КЛАССОВ

class_counts = df_analysis['label'].value_counts().sort_index()
class_names = ['Негативные (1-2)', 'Нейтральные (3)', 'Позитивные (4-5)']

plt.figure(figsize=(8, 6))
plt.bar(class_names, class_counts, color=['red', 'gray', 'green'])
plt.title('Распределение классов тональности в датасете', fontsize=14)
plt.xlabel('Класс тональности', fontsize=12)
plt.ylabel('Количество отзывов', fontsize=12)

for i, v in enumerate(class_counts):
    plt.text(i, v + 500, str(v), ha='center', fontsize=11)

plt.tight_layout()
plt.show()

#  АНАЛИЗ ДЛИНЫ ТЕКСТОВ

df_analysis['text_length'] = df_analysis['text'].astype(str).str.len()

plt.figure(figsize=(10, 6))
plt.hist(df_analysis['text_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Распределение длин текстов отзывов', fontsize=14)
plt.xlabel('Длина текста (количество символов)', fontsize=12)
plt.ylabel('Количество отзывов', fontsize=12)
plt.axvline(df_analysis['text_length'].mean(), color='red', linestyle='dashed', linewidth=2, 
            label=f'Среднее: {df_analysis["text_length"].mean():.0f}')
plt.axvline(df_analysis['text_length'].median(), color='green', linestyle='dashed', linewidth=2, 
            label=f'Медиана: {df_analysis["text_length"].median():.0f}')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nМинимальная длина: {df_analysis['text_length'].min()}")
print(f"Максимальная длина: {df_analysis['text_length'].max()}")
print(f"Средняя длина: {df_analysis['text_length'].mean():.0f}")
print(f"Медианная длина: {df_analysis['text_length'].median():.0f}")
