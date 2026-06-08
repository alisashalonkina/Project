#3.	ПЕРВИЧНЫЙ АНАЛИЗ НАБОРА ДАННЫХ С ИЗОБРАЖЕНИЯМИ
# Код был написан в Google Colab
# Основные переменные и монтаж google drive
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# Путь к датасету
DATASET_PATH = "/content/drive/MyDrive/archive/road_damage_dataset"

# Словарь классов
CLASS_NAMES = {
    0: "Pothole",
    1: "Crocodile cracking",
    2: "Lateral cracking",
    3: "Longitudinal cracking",
    4: "Sewer manhole"
}

# Получаем списки файлов
all_files = os.listdir(DATASET_PATH)
image_files = sorted([f for f in all_files if f.endswith('.png')])
txt_files = sorted([f for f in all_files if f.endswith('.txt')])

print("Google Drive смонтирован. Основные переменные созданы.")
print(f"Путь к датасету: {DATASET_PATH}")
print(f"Изображений: {len(image_files)}")
print(f"Аннотаций: {len(txt_files)}")
# Пункт 3.3.1 — АНАЛИЗ КОЛИЧЕСТВА И БАЛАНСА КЛАССОВ

import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

print("3.3.1. АНАЛИЗ КОЛИЧЕСТВА И БАЛАНСА КЛАССОВ")

# Проверяем соответствие изображений и аннотаций
image_names_no_ext = set([os.path.splitext(f)[0] for f in image_files])
txt_names_no_ext = set([os.path.splitext(f)[0] for f in txt_files])

missing_txt = image_names_no_ext - txt_names_no_ext
missing_img = txt_names_no_ext - image_names_no_ext

if missing_txt:
    print(f"ВНИМАНИЕ: {len(missing_txt)} изображений без аннотаций!")
else:
    print(" Каждому изображению соответствует файл аннотации.")

if missing_img:
    print(f"ВНИМАНИЕ: {len(missing_img)} аннотаций без изображений!")
else:
    print(" Каждому файлу аннотации соответствует изображение.")

# Подсчет объектов каждого класса
class_counts = Counter()
empty_annotations = 0
total_annotations = 0
objects_per_image = defaultdict(int)

for txt_file in txt_files:
    txt_path = os.path.join(DATASET_PATH, txt_file)
    img_name = os.path.splitext(txt_file)[0]

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    if len(lines) == 0:
        empty_annotations += 1
        objects_per_image[img_name] = 0
        continue

    total_annotations += len(lines)
    objects_per_image[img_name] = len(lines)

    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            class_id = int(float(parts[0]))
            class_counts[class_id] += 1

# Вывод результатов
print(f"\nОбщее количество размеченных дефектов: {total_annotations}")
print(f"Пустых аннотаций (дорог без дефектов): {empty_annotations}")
print(f"Доля пустых: {empty_annotations / len(txt_files) * 100:.1f}%")

print("\nРаспределение объектов по классам:")
for class_id in sorted(class_counts.keys()):
    print(f"  Класс {class_id} — {CLASS_NAMES[class_id]}: {class_counts[class_id]} шт.")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# График 1: Распределение классов
classes = [CLASS_NAMES[i] for i in sorted(class_counts.keys())]
counts = [class_counts[i] for i in sorted(class_counts.keys())]
colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

bars = axes[0].bar(classes, counts, color=colors, edgecolor='black', linewidth=1.2)
axes[0].set_title('Распределение классов дефектов дорожного покрытия', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Класс дефекта', fontsize=12)
axes[0].set_ylabel('Количество объектов', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# График 2: Соотношение изображений с дефектами и без
labels_pie = ['С дефектами', 'Без дефектов (пустые)']
sizes_pie = [len(txt_files) - empty_annotations, empty_annotations]
colors_pie = ['#66b3ff', '#ff9999']
explode_pie = (0, 0.1)

axes[1].pie(sizes_pie, explode=explode_pie, labels=labels_pie, colors=colors_pie,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Соотношение изображений с дефектами и без', fontsize=14, fontweight='bold');
# Пункт 3.3.2 — ПРИМЕРЫ ТИПИЧНЫХ ИЗОБРАЖЕНИЙ

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("3.3.2. ПРИМЕРЫ ТИПИЧНЫХ ИЗОБРАЖЕНИЙ")

def draw_yolo_boxes(image_path, label_path, class_names):
    """Рисует bounding boxes на изображении по YOLO-аннотации."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    if not os.path.exists(label_path):
        return img
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    colors = [
        (255, 0, 0),    # Синий — Pothole
        (0, 255, 0),    # Зеленый — Crocodile cracking
        (0, 0, 255),    # Красный — Lateral cracking
        (255, 255, 0),  # Желтый — Longitudinal cracking
        (255, 0, 255),  # Фиолетовый — Sewer manhole
    ]
    
    img_with_boxes = img.copy()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        class_id = int(float(parts[0]))
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_width = float(parts[3]) * w
        box_height = float(parts[4]) * h
        
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        color = colors[class_id % len(colors)]
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
        
        label = f"Class {class_id}: {class_names[class_id].split(' (')[0]}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_with_boxes, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
        cv2.putText(img_with_boxes, label, (x1 + 2, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_with_boxes

# Находим аннотации с объектами
txt_files_with_objects = []
for txt_file in txt_files:
    txt_path = os.path.join(DATASET_PATH, txt_file)
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) > 0:
        txt_files_with_objects.append(txt_file)

print(f"Количество изображений с объектами: {len(txt_files_with_objects)}")

# Выбираем 4 случайных изображения
np.random.seed(42)
sample_files = np.random.choice(txt_files_with_objects, size=min(4, len(txt_files_with_objects)), replace=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, txt_file in enumerate(sample_files):
    img_name = os.path.splitext(txt_file)[0] + '.png'
    img_path = os.path.join(DATASET_PATH, img_name)
    txt_path = os.path.join(DATASET_PATH, txt_file)
    
    img_with_boxes = draw_yolo_boxes(img_path, txt_path, CLASS_NAMES)
    
    if img_with_boxes is not None:
        axes[idx].imshow(img_with_boxes)
        axes[idx].set_title(f'Изображение: {img_name}', fontsize=11, fontweight='bold')
        axes[idx].axis('off')
    else:
        axes[idx].text(0.5, 0.5, 'Ошибка загрузки', ha='center', va='center')
        axes[idx].axis('off')

plt.suptitle('Примеры изображений с визуализированной разметкой', 
             fontsize=14, fontweight='bold', y=1.01);
# Пункт 3.3.3 — ОЦЕНКА КАЧЕСТВА ИЗОБРАЖЕНИЙ

import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


print("3.3.3. ОЦЕНКА КАЧЕСТВА ИЗОБРАЖЕНИЙ")

image_sizes = []
corrupted_files = []

for img_file in image_files:
    img_path = os.path.join(DATASET_PATH, img_file)
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            image_sizes.append({
                'filename': img_file,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}"
            })
    except Exception as e:
        corrupted_files.append(img_file)
        print(f"Ошибка при открытии {img_file}: {e}")

print(f"\nУспешно проанализировано: {len(image_sizes)} изображений")
print(f"Поврежденных файлов: {len(corrupted_files)}")

df_sizes = pd.DataFrame(image_sizes)

print("\nУникальные разрешения и их количество:")
resolution_counts = df_sizes['resolution'].value_counts()
for res, count in resolution_counts.items():
    print(f"  {res}: {count} изображений ({count/len(df_sizes)*100:.1f}%)")

print(f"\nМинимальная ширина: {df_sizes['width'].min()} пикселей")
print(f"Максимальная ширина: {df_sizes['width'].max()} пикселей")
print(f"Минимальная высота: {df_sizes['height'].min()} пикселей")
print(f"Максимальная высота: {df_sizes['height'].max()} пикселей")

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

res_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
axes[0].bar(resolution_counts.index, resolution_counts.values, 
           color=res_colors[:len(resolution_counts)], edgecolor='black', linewidth=1.2)
axes[0].set_title('Распределение разрешений', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Разрешение', fontsize=11)
axes[0].set_ylabel('Количество', fontsize=11)
for i, (res, count) in enumerate(resolution_counts.items()):
    axes[0].text(i, count + max(resolution_counts.values)*0.02, str(count), 
                ha='center', fontweight='bold')

axes[1].scatter(df_sizes['width'], df_sizes['height'], alpha=0.5, s=30, 
               c='steelblue', edgecolors='black', linewidth=0.5)
axes[1].set_title('Ширина vs Высота', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Ширина (пиксели)', fontsize=11)
axes[1].set_ylabel('Высота (пиксели)', fontsize=11)
axes[1].axhline(y=512, color='red', linestyle='--', alpha=0.7, label='Мин. 512px')
axes[1].axvline(x=512, color='red', linestyle='--', alpha=0.7)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(df_sizes['width'], alpha=0.7, label='Ширина', bins=20, color='steelblue', edgecolor='black')
axes[2].hist(df_sizes['height'], alpha=0.7, label='Высота', bins=20, color='coral', edgecolor='black')
axes[2].set_title('Гистограмма размеров', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Пиксели', fontsize=11)
axes[2].set_ylabel('Частота', fontsize=11)
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Анализ качества и размеров изображений', 
             fontsize=14, fontweight='bold', y=1.02);
# Пункт 3.3.4 — АНАЛИЗ АННОТАЦИЙ

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("3.3.4. АНАЛИЗ АННОТАЦИЙ")

# Собираем данные о количестве объектов на изображении
objects_per_image = defaultdict(int)

for txt_file in txt_files:
    txt_path = os.path.join(DATASET_PATH, txt_file)
    img_name = os.path.splitext(txt_file)[0]
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    objects_per_image[img_name] = len(lines)

objects_per_image_values = list(objects_per_image.values())

# Статистика
print(f"Среднее количество объектов на изображении: {np.mean(objects_per_image_values):.2f}")
print(f"Медианное количество объектов: {np.median(objects_per_image_values):.1f}")
print(f"Максимум объектов на одном изображении: {max(objects_per_image_values)}")
print(f"Минимум объектов: {min(objects_per_image_values)}")
print(f"Стандартное отклонение: {np.std(objects_per_image_values):.2f}")

# Распределение по диапазонам
print("\nРаспределение изображений по количеству объектов:")
ranges = [(0, 0), (1, 3), (4, 6), (7, 10), (11, 20), (21, 100)]
for low, high in ranges:
    count = sum(1 for x in objects_per_image_values if low <= x <= high)
    print(f"  {low}-{high} объектов: {count:>4} изобр. ({count/len(objects_per_image_values)*100:5.1f}%)")

# Топ-10 изображений с наибольшим количеством объектов
print("\nТоп-10 изображений с наибольшим количеством дефектов:")
top_objects = sorted(objects_per_image.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (img_name, count) in enumerate(top_objects, 1):
    print(f"  {i:2}. {img_name}.png: {count} дефектов")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

max_display = 25
filtered_values = [x for x in objects_per_image_values if x <= max_display]
outliers_count = sum(1 for x in objects_per_image_values if x > max_display)

axes[0].hist(filtered_values, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axvline(np.mean(objects_per_image_values), color='red', linestyle='--', 
                linewidth=2, label=f'Среднее: {np.mean(objects_per_image_values):.1f}')
axes[0].axvline(np.median(objects_per_image_values), color='green', linestyle='--', 
                linewidth=2, label=f'Медиана: {np.median(objects_per_image_values):.1f}')
axes[0].set_title(f'Распределение количества дефектов на изображении\n(показаны ≤ {max_display}, выбросов: {outliers_count})', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Количество дефектов', fontsize=11)
axes[0].set_ylabel('Количество изображений', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

bp = axes[1].boxplot(objects_per_image_values, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
axes[1].set_title('Ящик с усами: количество дефектов', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Количество дефектов', fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

q1 = np.percentile(objects_per_image_values, 25)
q3 = np.percentile(objects_per_image_values, 75)
med = np.median(objects_per_image_values)
stats_text = f"Мин: {min(objects_per_image_values)}\nQ1: {q1:.1f}\nМедиана: {med:.1f}\nСреднее: {np.mean(objects_per_image_values):.1f}\nQ3: {q3:.1f}\nМакс: {max(objects_per_image_values)}"
axes[1].text(1.15, med, stats_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Распределение количества дефектов на одном изображении', 
             fontsize=14, fontweight='bold', y=1.02);

# Пункт 3.3.5 — ОЦЕНКА КАЧЕСТВА РАЗМЕТКИ

import os
import matplotlib.pyplot as plt

print("3.3.5. ОЦЕНКА КАЧЕСТВА РАЗМЕТКИ")

errors = {
    'invalid_class_id': [],
    'coordinate_out_of_range': [],
    'zero_dimensions': [],
    'malformed_lines': []
}

print(f"Проверка {len(txt_files)} файлов аннотаций...")

for i, txt_file in enumerate(txt_files):
    # Простой прогресс каждые 500 файлов
    if (i + 1) % 500 == 0 or (i + 1) == len(txt_files):
        print(f"  Проверено: {i + 1}/{len(txt_files)} файлов")
    
    txt_path = os.path.join(DATASET_PATH, txt_file)
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        if len(parts) != 5:
            errors['malformed_lines'].append({
                'file': txt_file,
                'line': line_num,
                'content': line,
                'reason': f'Элементов: {len(parts)} (нужно 5)'
            })
            continue
        
        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as e:
            errors['malformed_lines'].append({
                'file': txt_file,
                'line': line_num,
                'content': line,
                'reason': f'Ошибка: {e}'
            })
            continue
        
        # Проверка class_id
        if class_id not in CLASS_NAMES:
            errors['invalid_class_id'].append({
                'file': txt_file,
                'line': line_num,
                'class_id': class_id
            })
        
        # Проверка координат
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            errors['coordinate_out_of_range'].append({
                'file': txt_file,
                'line': line_num,
                'x_center': x_center,
                'y_center': y_center
            })
        
        if not (0 <= width <= 1 and 0 <= height <= 1):
            errors['coordinate_out_of_range'].append({
                'file': txt_file,
                'line': line_num,
                'width': width,
                'height': height,
                'reason': 'Размеры вне [0, 1]'
            })
        
        # Проверка на нулевые размеры
        if width <= 0 or height <= 0:
            errors['zero_dimensions'].append({
                'file': txt_file,
                'line': line_num,
                'width': width,
                'height': height
            })

print("Проверка завершена!")

# Вывод результатов
print("\nРезультаты проверки:")
print("{:<50} {:<15} {:<10}".format("Тип проверки", "Результат", "Ошибок"))
print("-" * 80)

checks = [
    ("Корректность class_id (0-4)", len(errors['invalid_class_id'])),
    ("Координаты рамок в диапазоне [0, 1]", len(errors['coordinate_out_of_range'])),
    ("Ширина и высота рамки > 0", len(errors['zero_dimensions'])),
    ("Корректность формата строк", len(errors['malformed_lines']))
]

total_errors = sum(c[1] for c in checks)

for check_name, error_count in checks:
    status = "Пройдена" if error_count == 0 else f"Ошибок: {error_count}"
    print("{:<50} {:<15} {:<10}".format(check_name, status, error_count))

print("-" * 80)
print("{:<50} {:<15} {:<10}".format("ИТОГО", "", total_errors))

if total_errors > 0:
    print(f"\nОбнаружено ошибок: {total_errors}")
    for error_type, error_list in errors.items():
        if error_list:
            print(f"\n  {error_type}: {len(error_list)} sht.")
            for err in error_list[:3]:
                print(f"    - {err.get('file', '')}, stroka {err.get('line', '')}")
                print(f"      Причина: {err.get('reason', '')}")
else:
    print("\nВсе проверки пройдены! Формальных ошибок не обнаружено.")

# Визуализация
fig, ax = plt.subplots(figsize=(10, 6))

check_names = [c[0] for c in checks]
error_counts_list = [c[1] for c in checks]

bar_colors = ['#2ecc71' if count == 0 else '#e74c3c' for count in error_counts_list]
bars = ax.barh(check_names, error_counts_list, color=bar_colors, edgecolor='black', linewidth=1.5)

ax.set_title('Результаты проверки качества разметки', fontsize=14, fontweight='bold')
ax.set_xlabel('Количество ошибок', fontsize=12)

for bar, count in zip(bars, error_counts_list):
    label = '0 ошибок' if count == 0 else f'{count} ошибок'
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            label, va='center', fontweight='bold', fontsize=11)

ax.set_xlim(0, max(max(error_counts_list) + 2, 5))
ax.grid(axis='x', alpha=0.3)
