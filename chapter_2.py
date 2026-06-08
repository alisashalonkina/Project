#2.	ПЕРВИЧНЫЙ АНАЛИЗ НАБОРА ДАННЫХ С ВРЕМЕННЫМИ РЯДАМИ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# ЭТАП 1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО С ДАННЫМИ

# Загрузка данных (для Google Colab)
# from google.colab import files
# uploaded = files.upload()

df = pd.read_csv('5.csv', sep=';', parse_dates=['datetime'], index_col='datetime')

print("=" * 70)
print("ЭТАП 1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО С ДАННЫМИ")
print("=" * 70)

print("\nПервые 5 строк DataFrame:")
print(df.head())

print(f"\nРазмерность данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"\nТипы данных:\n{df.dtypes}")

# Частота дискретизации (более надежный способ)
time_diffs = df.index.to_series().diff().dt.total_seconds()
print(f"\nЧастота дискретизации: {time_diffs.mode()[0]} секунд")
print(f"Минимальная разница: {time_diffs.min()} с, Максимальная разница: {time_diffs.max()} с")

# Список измерительных каналов (без меток аномалий)
sensor_columns = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
                  'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']


# ЭТАП 2. ВИЗУАЛИЗАЦИЯ ИСХОДНЫХ ДАННЫХ

print("\n" + "=" * 70)
print("ЭТАП 2. ВИЗУАЛИЗАЦИЯ ИСХОДНЫХ ДАННЫХ")
print("=" * 70)

# Построение графиков всех каналов
fig, axes = plt.subplots(nrows=len(sensor_columns), ncols=1, figsize=(15, 16), sharex=True)

for i, col in enumerate(sensor_columns):
    # Основной график
    axes[i].plot(df.index, df[col], color='steelblue', linewidth=0.8, alpha=0.7)

    # Подсветка интервалов коллективных аномалий (changepoint == 1)
    axes[i].fill_between(df.index, axes[i].get_ylim()[0], axes[i].get_ylim()[1],
                         where=df['changepoint'] == 1, color='red', alpha=0.25)

    # Выделение точечных аномалий (anomaly == 1)
    point_anomalies = df[df['anomaly'] == 1]
    if not point_anomalies.empty:
        axes[i].scatter(point_anomalies.index, point_anomalies[col],
                       color='green', s=15, alpha=0.7, zorder=5)

    axes[i].set_ylabel(col, fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.3)
    axes[i].tick_params(axis='x', rotation=45)

axes[-1].set_xlabel('Время', fontsize=12)
plt.suptitle('Многомерный временной ряд с выделением аномалий (файл valve1/5.csv)',
             fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

# Информация об аномальных интервалах
anomaly_mask = df['changepoint'] == 1
if anomaly_mask.any():
    # Находим непрерывные интервалы
    anomaly_start = df.index[anomaly_mask & ~anomaly_mask.shift(1).fillna(False)]
    anomaly_end = df.index[anomaly_mask & ~anomaly_mask.shift(-1).fillna(False)]
    print("\nНайденные интервалы коллективных аномалий:")
    for start, end in zip(anomaly_start, anomaly_end):
        print(f"  {start} — {end}")


# ЭТАП 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ

print("\n" + "=" * 70)
print("ЭТАП 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ")
print("=" * 70)

# Расчет статистик для измерительных каналов
stats_df = df[sensor_columns].describe().T
stats_df['median'] = df[sensor_columns].median()
stats_df['Q1'] = df[sensor_columns].quantile(0.25)
stats_df['Q3'] = df[sensor_columns].quantile(0.75)
stats_df['IQR'] = stats_df['Q3'] - stats_df['Q1']

print("\nСтатистические характеристики каналов:")
print(stats_df[['mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max', 'IQR']].round(4))

# Дополнительно: статистика для нормального участка (без аномалий)
normal_data = df[df['changepoint'] == 0]
if len(normal_data) > 0:
    print("\nСтатистика для НОРМАЛЬНОГО участка (changepoint=0):")
    normal_stats = normal_data[sensor_columns].median()
    print(pd.DataFrame(normal_stats).T.round(4))


# ЭТАП 4. АНАЛИЗ ПРОПУСКОВ И ВЫБРОСОВ

print("\n" + "=" * 70)
print("ЭТАП 4. АНАЛИЗ ПРОПУСКОВ И ВЫБРОСОВ")
print("=" * 70)

# Пропуски
print("\nКоличество пропущенных значений:")
print(df.isnull().sum())

# Диаграмма размаха (более информативна, чем правило 3σ для ненормальных данных)
plt.figure(figsize=(14, 6))
sns.boxplot(data=df[sensor_columns])
plt.title('Диаграмма размаха для каждого канала (файл 5.csv)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ЭТАП 5. АНАЛИЗ ДИАПАЗОНОВ ЗНАЧЕНИЙ

print("\n" + "=" * 70)
print("ЭТАП 5. АНАЛИЗ ДИАПАЗОНОВ ЗНАЧЕНИЙ")
print("=" * 70)

print("\nДИАПАЗОНЫ ЗНАЧЕНИЙ КАНАЛОВ:")
print("-" * 60)
for col in sensor_columns:
    print(f"{col:25s}: [{df[col].min():8.4f}, {df[col].max():8.4f}]  "
          f"(размах = {df[col].max() - df[col].min():8.4f})")

# Гистограммы распределений
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(sensor_columns):
    axes[i].hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].axvline(df[col].mean(), color='red', linestyle='--',
                    label=f'Ср. = {df[col].mean():.2f}')
    axes[i].axvline(df[col].median(), color='green', linestyle='--',
                    label=f'Мед. = {df[col].median():.2f}')
    axes[i].set_title(f'Распределение {col}', fontsize=10)
    axes[i].set_xlabel('Значение', fontsize=9)
    axes[i].set_ylabel('Частота', fontsize=9)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Гистограммы распределения значений каналов', fontsize=14)
plt.tight_layout()
plt.show()

# ЭТАП 6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ

print("\n" + "=" * 70)
print("ЭТАП 6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 70)

# Матрица корреляции
corr_matrix = df[sensor_columns].corr()

# Тепловая карта (полная, без маски)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, center=0)
plt.title('Матрица корреляции Пирсона между каналами данных', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Наиболее сильные корреляции
print("\nНАИБОЛЕЕ СИЛЬНЫЕ ПОЛОЖИТЕЛЬНЫЕ КОРРЕЛЯЦИИ:")
print("-" * 50)
corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
corr_flat = corr_flat.unstack().dropna().sort_values(ascending=False)
for (row, col), val in corr_flat.head(5).items():
    print(f"  {row} — {col}: {val:.3f}")

print("\nНАИБОЛЕЕ СИЛЬНЫЕ ОТРИЦАТЕЛЬНЫЕ КОРРЕЛЯЦИИ:")
print("-" * 50)
for (row, col), val in corr_flat.tail(5).items():
    print(f"  {row} — {col}: {val:.3f}")


# ЭТАП 7. ПОИСК И АНАЛИЗ ШУМОВ (С ОЦЕНКОЙ КОНТРАСТА)

print("\n" + "=" * 70)
print("ЭТАП 7. ПОИСК И АНАЛИЗ ШУМОВ")
print("=" * 70)
print("АНАЛИЗ ШУМОВ ДЛЯ КАНАЛА: Current")
print("="*70)

signal_col = 'Current'

# 7.1. ПОДГОТОВКА ДАННЫХ (чистый нормальный участок)

# Фильтруем: исключаем и коллективные, и точечные аномалии
clean_mask = (df['changepoint'] == 0) & (df['anomaly'] == 0)
series_clean = df.loc[clean_mask, signal_col].reset_index(drop=True)

print(f"\n--- ИСХОДНЫЕ ДАННЫЕ ---")
print(f"Всего записей:                         {len(df)}")
print(f"Нормальный режим (changepoint=0):      {len(df[df['changepoint'] == 0])} записей")
print(f"Чистый нормальный (changepoint=0 и anomaly=0): {len(series_clean)} записей")

# Выбираем период декомпозиции
period = min(20, len(series_clean) // 3 if len(series_clean) > 20 else 10)
print(f"Период декомпозиции: {period} сек")


# 7.2. ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА

print("\n" + "="*70)
print("ДЕКОМПОЗИЦИЯ НА ЧИСТОМ НОРМАЛЬНОМ УЧАСТКЕ")
print("(changepoint = 0 И anomaly = 0)")
print("="*70)

decomp_clean = seasonal_decompose(series_clean, model='additive', period=period)
trend_clean = decomp_clean.trend
seasonal_clean = decomp_clean.seasonal
resid_clean = decomp_clean.resid

# Визуализация декомпозиции
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Исходный ряд
axes[0].plot(series_clean.index, series_clean, color='blue', linewidth=0.8)
axes[0].set_title(f'Исходный временной ряд ({signal_col}) - чистый нормальный режим', fontsize=12)
axes[0].set_ylabel('Значение (А)', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Тренд
axes[1].plot(trend_clean.index, trend_clean, color='green', linewidth=0.8)
axes[1].set_title('Трендовая компонента', fontsize=12)
axes[1].set_ylabel('Значение (А)', fontsize=10)
axes[1].grid(True, alpha=0.3)

# Сезонность
axes[2].plot(seasonal_clean.index, seasonal_clean, color='orange', linewidth=0.8)
axes[2].set_title(f'Сезонная компонента (период = {period} сек)', fontsize=12)
axes[2].set_ylabel('Значение (А)', fontsize=10)
axes[2].grid(True, alpha=0.3)

# Остатки (шум)
axes[3].plot(resid_clean.index, resid_clean, color='red', linewidth=0.5)
axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[3].set_title('Шумовая компонента (остатки)', fontsize=12)
axes[3].set_xlabel('Время (отсчёты)', fontsize=12)
axes[3].set_ylabel('Значение (А)', fontsize=10)
axes[3].grid(True, alpha=0.3)

plt.suptitle(f'Декомпозиция временного ряда: {signal_col} (чистый нормальный режим)', fontsize=14)
plt.tight_layout()
plt.show()


# 7.3. РАСЧЕТ ОТНОШЕНИЯ СИГНАЛ/ШУМ (SNR)

signal_comp = trend_clean + seasonal_clean
signal_vals = signal_comp.dropna()
noise_vals = resid_clean.dropna()

signal_var = signal_vals.var()
noise_var = noise_vals.var()
snr = 10 * np.log10(signal_var / noise_var)

print(f"\n--- РЕЗУЛЬТАТЫ ДЛЯ ЧИСТОГО НОРМАЛЬНОГО УЧАСТКА ---")
print(f"Дисперсия сигнала:   {signal_var:.6f}")
print(f"Дисперсия шума:      {noise_var:.6f}")
print(f"SNR:                 {snr:.2f} дБ")

# Интерпретация SNR
if snr > 20:
    quality = "Отлично"
    comment = "Шум практически незаметен, данные очень чистые."
elif snr > 10:
    quality = "Хорошо"
    comment = "Шум присутствует, но сигнал доминирует. Модели будут работать стабильно."
elif snr > 0:
    quality = "Удовлетворительно"
    comment = "Сигнал и шум сравнимы по мощности. Данные пригодны для анализа."
else:
    quality = "Плохо"
    comment = "Шум сильнее сигнала. Данные требуют фильтрации."

print(f"Качественная оценка: {quality}")
print(f"Вывод: {comment}")


# 7.4. АНАЛИЗ РАСПРЕДЕЛЕНИЯ ШУМА

print("\n" + "="*70)
print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ШУМА (чистый нормальный режим)")
print("="*70)

resid_vals = resid_clean.dropna()

# Гистограмма и Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма с кривой плотности
sns.histplot(resid_vals, kde=True, bins=50, color='purple', alpha=0.6, ax=axes[0])
axes[0].set_title('Распределение шумовой компоненты (остатки)\nчистый нормальный режим', fontsize=12)
axes[0].set_xlabel('Значение остатка (А)', fontsize=10)
axes[0].set_ylabel('Частота', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(resid_vals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q plot остатков (проверка на нормальность)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Статистические характеристики
print(f"\n--- СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ ОСТАТКОВ ---")
print(f"Среднее значение:       {resid_vals.mean():.6f}")
print(f"Медиана:                {resid_vals.median():.6f}")
print(f"Стандартное отклонение: {resid_vals.std():.6f}")
print(f"Асимметрия (skewness):  {resid_vals.skew():.4f}")
print(f"Эксцесс (kurtosis):     {resid_vals.kurtosis():.4f}")

# Тест Шапиро-Уилка
if len(resid_vals) <= 5000:
    shapiro_stat, shapiro_p = stats.shapiro(resid_vals)
    print(f"\n--- ТЕСТ ШАПИРО-УИЛКА НА НОРМАЛЬНОСТЬ ---")
    print(f"Статистика: {shapiro_stat:.4f}")
    print(f"p-value:    {shapiro_p:.4e}")
    if shapiro_p > 0.05:
        print("Вывод: Распределение НЕ отличается от нормального (p > 0.05)")
        print("→ Шум является «белым шумом», данные хорошего качества.")
    else:
        print("Вывод: Распределение ОТЛИЧАЕТСЯ от нормального (p < 0.05)")
        print("→ Шум имеет неслучайную структуру или тяжелые хвосты.")

# 7.5. ИТОГОВЫЙ ВЫВОД

print("\n" + "="*70)
print("ИТОГОВЫЙ ВЫВОД ПО АНАЛИЗУ ШУМОВ")
print("="*70)

if snr > 10:
    print(" SNR на чистом нормальном участке > 10 дБ — качество ХОРОШЕЕ")
    print(" Фильтрация данных НЕ ТРЕБУЕТСЯ")
    print(" Данные пригодны для задачи обнаружения аномалий")
elif snr > 0:
    print(" SNR на чистом нормальном участке 0–10 дБ — качество УДОВЛЕТВОРИТЕЛЬНОЕ")
    print(" Рекомендуется лёгкая фильтрация (скользящее среднее)")
else:
    print(" SNR на чистом нормальном участке < 0 дБ — качество ПЛОХОЕ")
    print(" Требуется серьёзная фильтрация (медианный фильтр, вейвлеты)")

print("\nРекомендация по масштабированию: использовать RobustScaler")
