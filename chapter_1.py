import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из файла
try:
    df = pd.read_csv('SKATERS.csv')
    print("Данные успешно загружены!")
except FileNotFoundError:
    print("Ошибка: Файл SKATERS.csv не найден.")

# Выведем первые 5 строк таблицы
print(df.head())
# 1.3.1.	Визуализация распределения признаков
numeric_cols = ['games', 'goals', 'assists', 'time'] # Числовые признаки, которые будем анализировать

plt.figure(figsize=(12, 10)) # Размер графика

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    # Строим гистограмму, kde=True добавляет сглаживающую линию (оценку плотности распределения)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Распределение признака: {col}')
    plt.xlabel(f'Признак: {col}')
    plt.ylabel('Количество игроков')

plt.tight_layout() # Чтобы подписи не накладывались друг на друга
plt.show()
# 1.3.2.	Визуализация признаков
# Считаем количество игроков на каждой уникальной позиции
position_counts = df['position'].value_counts()

plt.figure(figsize=(10, 6))# Размер графика
# Строим круговую диаграмму
plt.pie(position_counts, labels=position_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Распределение игроков по позициям')
plt.show()

# Выводим количество в виде текста

print("Количество игроков по позициям:")
print(position_counts)
# 1.3.3.	Анализ на наличие пропусков в данных
# Создаем таблицу с информацией о пропусках
missing_data = pd.DataFrame({
    'Признак': df.columns,
    'Количество пропусков': df.isnull().sum(),
    'Доля пропусков, %': (df.isnull().sum() / len(df)) * 100
})

# Сортируем по количеству пропусков для наглядности
missing_data = missing_data.sort_values(by='Количество пропусков', ascending=False)

# Выводим результат на экран
print(missing_data.to_string(index=False))
# 1.3.4. Кореляционный анализ
# Первая тепловая карта: карьерные показатели (игровая статистика)
# Выбираем признаки, отражающие результативность и опыт
performance_cols = ['games', 'goals', 'assists', 'time']
corr_performance = df[performance_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_performance, annot=True, cmap='coolwarm', fmt='.2f',
            square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Тепловая карта корреляции: игровые показатели', fontsize=14)
plt.show()

# Вторая тепловая карта: временные характеристики карьеры
# Выбираем признаки, связанные с периодами выступлений
temporal_cols = ['first', 'last', 'games']
# Рассчитываем продолжительность карьеры как разницу между последним и первым сезоном
df['career_length'] = df['last'] - df['first']
temporal_cols_with_length = ['first', 'last', 'career_length', 'games']
corr_temporal = df[temporal_cols_with_length].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_temporal, annot=True, cmap='coolwarm', fmt='.2f',
            square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Тепловая карта корреляции: временные характеристики карьеры', fontsize=14)
plt.show()

# Вывод пояснений
print("\nАнализ первой тепловой карты (игровые показатели):")
print("- Корреляция между goals и assists: {:.2f}".format(corr_performance.loc['goals', 'assists']))
print("- Корреляция между games и time: {:.2f}".format(corr_performance.loc['games', 'time']))

print("\nАнализ второй тепловой карты (временные характеристики):")
print("- Корреляция между first и last: {:.2f}".format(corr_temporal.loc['first', 'last']))
print("- Корреляция между career_length и games: {:.2f}".format(corr_temporal.loc['career_length', 'games']))

# 1.3.5.	Устранение дубликатов
# Проверяем количество полных дубликатов
num_duplicates = df.duplicated().sum()
print(f"Количество найденных полных дубликатов: {num_duplicates}")
# 1.3.6.	Анализ и обработка выбросов
# Выбираем признаки для анализа выбросов
# Анализируем ключевые статистические показатели: матчи, голы, передачи
cols_for_boxplot = ['games', 'goals', 'assists']

# Устанавливаем размер графика (ширина 15 дюймов, высота 6)
plt.figure(figsize=(15, 6))

# В цикле создаем 3 подграфика в одной строке
for i, col in enumerate(cols_for_boxplot, 1):
    plt.subplot(1, 3, i)  # 1 строка, 3 колонки, текущий график номер i
    sns.boxplot(y=df[col])  # Строим вертикальный ящик с усами
    plt.title(f'Ящик с усами для признака: {col}')
    plt.ylabel(col)  # Подписываем ось Y

# Функция tight_layout автоматически регулирует отступы,
# чтобы подписи не накладывались друг на друга
plt.tight_layout()
# Отображаем график
plt.show()
# Создаем новую колонку 'goals_log'
# Функция np.log1p(x) вычисляет натуральный логарифм от (x + 1)
# Добавление единицы (+1) необходимо, чтобы избежать ошибки вычисления log(0)
# для игроков, которые не забили ни одного гола
df['goals_log'] = np.log1p(df['goals'])

# Строим два графика рядом для наглядного сравнения
plt.figure(figsize=(12, 5))

# Левый график: исходное распределение
plt.subplot(1, 2, 1)
sns.histplot(df['goals'], kde=True, bins=30)
plt.title('Распределение голов (до преобразования)')
plt.xlabel('Количество голов')
plt.ylabel('Количество игроков')

# Правый график: распределение после логарифмирования
plt.subplot(1, 2, 2)
sns.histplot(df['goals_log'], kde=True, bins=30)
plt.title('Распределение голов (после логарифмирования)')
plt.xlabel('Логарифм количества голов')
plt.ylabel('Количество игроков')

plt.tight_layout()
plt.show()
# 1.3.7. Фильтрация данных
print("\n" + "="*60)
print("ФИЛЬТРАЦИЯ 1: Опытные игроки (500+ матчей)")
print("="*60)

# Фильтр 1: Игроки, сыгравшие 500 и более матчей
filter1_players = df[df['games'] >= 500]

print(f"Общее количество игроков в исходном наборе: {len(df)}")
print(f"Количество игроков, сыгравших 500 и более матчей: {len(filter1_players)}")
print(f"Доля от общего числа игроков: {(len(filter1_players) / len(df)) * 100:.2f}%")
print("\nПервые 5 записей об опытных игроках (>= 500 матчей):")
print(filter1_players[['name', 'games', 'goals', 'assists']].head().to_string(index=False))


print("\n" + "="*60)
print("ФИЛЬТРАЦИЯ 2: Результативные защитники (100+ голов)")
print("="*60)

# Фильтр 2: Защитники (позиция 'D'), забившие более 100 голов
# Обратите внимание: в данных позиция может быть указана как 'D' или 'D/RW' и т.д.
# Используем .str.contains для поиска защитников в том числе с комбинированными позициями
filter2_players = df[df['position'].str.contains('D', na=False) & (df['goals'] > 100)]

print(f"Количество защитников, забивших более 100 голов: {len(filter2_players)}")
print(f"Доля от общего числа игроков: {(len(filter2_players) / len(df)) * 100:.4f}%")
print("\nПервые 10 самых результативных защитников (сортировка по голам):")
# Сортируем по голам и выводим топ-10
top_defensemen = filter2_players.nlargest(10, 'goals')[['name', 'position', 'games', 'goals', 'assists']]
print(top_defensemen.to_string(index=False))


print("\n" + "="*60)
print("ФИЛЬТРАЦИЯ 3: Игроки современной эпохи, не попавшие в Зал славы")
print("="*60)

# Фильтр 3: Игроки, завершившие карьеру после 2000 года (last >= 2000),
#          не являющиеся действующими (active = False) и не попавшие в Зал славы (hof = False)
filter3_players = df[(df['last'] >= 2000) & (df['active'] == False) & (df['hof'] == False)]

print(f"Количество игроков, завершивших карьеру после 2000 г. и не попавших в Зал славы: {len(filter3_players)}")
print(f"Доля от общего числа игроков: {(len(filter3_players) / len(df)) * 100:.2f}%")
print("\nПервые 10 игроков (сортировка по году окончания карьеры):")
# Сортируем по году окончания карьеры (по убыванию)
top_recent = filter3_players.nlargest(10, 'last')[['name', 'position', 'first', 'last', 'games', 'goals', 'assists', 'hof']]
print(top_recent.to_string(index=False))
# 1.3.8.	Добавление шума
# Создаем копии исходных признаков с добавлением шума
# Параметр noise_std задает интенсивность шума (стандартное отклонение)
noise_std = 2.5

# Устанавливаем случайное зерно для воспроизводимости результатов
np.random.seed(42)

# Генерируем шум для каждого признака
noise_games = np.random.normal(0, noise_std, len(df))
noise_goals = np.random.normal(0, noise_std, len(df))
noise_assists = np.random.normal(0, noise_std, len(df))

# Добавляем шум к исходным данным, округляем и обнуляем отрицательные значения
df['games_noisy'] = np.maximum(0, np.round(df['games'] + noise_games)).astype(int)
df['goals_noisy'] = np.maximum(0, np.round(df['goals'] + noise_goals)).astype(int)
df['assists_noisy'] = np.maximum(0, np.round(df['assists'] + noise_assists)).astype(int)
# Выводим сравнительную таблицу
comparison_df = df[['games', 'games_noisy', 'goals', 'goals_noisy', 'assists', 'assists_noisy']].head()
print(comparison_df)
