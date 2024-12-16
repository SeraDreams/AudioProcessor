# Аудио Процессор: Многопоточный Конвертер и Организатор Файлов

## Обзор

Этот скрипт представляет собой мощный инструмент для пакетной обработки аудиофайлов с поддержкой:
- Конвертации между различными аудиоформатами
- Перемещения файлов
- Создания структурированного хранилища файлов
- Удаления дубликатов (настраиваемое по столбцу)
- Многопоточной обработки
- Сохранения результатов в CSV в реальном времени

## Возможности

### 🔄 Конвертация и Перемещение Файлов
- Поддержка конвертации между различными аудиоформатами с помощью FFmpeg.
- Возможность полного перемещения файлов из указанных CSV-файлов без конвертации.
- Настраиваемые параметры конвертации через FFmpeg. 
- Выбор между копированием и перемещением файлов.

### 📂 Структурированное Хранение
- Динамическое создание многоуровневой структуры директорий с гибкой конфигурацией уровней.
- Генерация уникальных имен файлов с использованием хеширования для предотвращения коллизий.
- Настраиваемая глубина и структура директорий (например, 16x16, 4x7x10x23, или любое другое число/список чисел).

### 🚀 Производительность
- Многопроцессорная обработка файлов для ускорения обработки больших объемов данных.
- Пакетная обработка с настраиваемым размером пакета для оптимизации производительности.
- Низкоуровневая оптимизация работы с файлами и использование tqdm для отображения прогресса.

### 🧹 Очистка Данных
- Автоматическое удаление дубликатов из CSV-файлов на основе указанного столбца. Возможность отключить удаление дубликатов.
- Подробная статистика обработки файлов, включая количество обработанных файлов каждого формата и удаленных дубликатов.
- Сохранение CSV результатов в режиме реального времени.

## Требования

- Python 3.7+
- FFmpeg
- Библиотеки:
 - `tqdm`
 - `pandas`
 - Стандартные библиотеки Python

## Установка

1. Установите Python 3.7 или выше.
2. Установите FFmpeg.
3. Установите зависимости:
    ```bash
    pip install tqdm pandas
    ```

## Использование

### Базовый Запуск

```bash
python audio_processor.py input1.csv input2.csv -o output_dataset
```

### Расширенные Параметры

```bash
python audio_processor.py input.csv \
    -o dataset \                                     # Выходная директория
    -oc processed_metadata.csv \                     # Имя выходного CSV файла
    -c .wav .mp3 \                                   # Форматы для конвертации
    -m .flac \                                       # Форматы для перемещения
    -t .mp3 \                                        # Целевое расширение
    -l 16 16 \                                       # Конфигурация уровней структуры директорий (2 уровня по 16 папок)
    --ffmpeg "-ar 16000 -b:a 64k -c:a libmp3lame"    # Параметры FFmpeg (в кавычках!)
    -bs 24 \                                         # Размер пакета файлов
    --cpu 6 \                                        # Количество используемых ядер
    -scm True \                                      # Копировать, а не перемещать
    -csr True \                                      # Сохранение CSV в реальном времени
    -di 0                                            # Удалять дубликаты по первому столбцу (индекс 0)
```

## Параметры Командной Строки

### Обязательные Параметры

- `csv_files`: Один или несколько путей к CSV-файлам.

### Опциональные Параметры

- `-o, --output`: Базовая директория для вывода.
- `-oc, --output-csv`: Имя выходного CSV-файла.
- `-hd, --headers`: Заголовки для выходного CSV.
- `-l, --levels`: Конфигурация уровней структуры директорий.
- `-c, --convert`: Форматы файлов для конвертации.
- `-m, --move`: Форматы файлов для перемещения.
- `-t, --target-ext`: Целевое расширение файла для конвертации.
- `-fm, --full-move`: CSV-файлы, для которых нужно полностью перемещать файлы.
- `--ffmpeg`: Пользовательские параметры FFmpeg (в кавычках!) Какие параметры есть у FFmpeg можно посмотреть в интернете.
- `-bs, --batch-size`: Размер пакета файлов.
- `--cpu`: Количество используемых ядер процессора.
- `-scm, --switch-copy-move`: Переключатель копирование/перемещение (True - перемещение, False - копирование).
- `-csr, --csv-save-realtime`: Сохранять CSV в реальном времени.
- `-di, --duplicate-index`: Индекс столбца для проверки на дубликаты (По умолчанию None - дубликаты не удаляются).


## Формат CSV-файла

Входные CSV-файлы должны иметь следующую структуру:
```
/path/to/audio1.wav|Привет, мир|3.5
/path/to/audio2.mp3|Доброе утро|2.7
```

- Разделитель: |
- Колонки:
1. Полный путь к аудиофайлу.
2. Транскрипция или текст.
3. Длительность аудио в секундах.

## Примеры

### Конвертация Аудио
```bash
python audio_processor.py dataset.csv -c .mp3 -t .wav
```
Конвертирует все MP3-файлы в WAV.

### Организация Больших Наборов Данных
```bash
python audio_processor.py train.csv validation.csv \
    -o speech_dataset \
    -l 32 \
    -c .wav .mp3 .flac
```
Обрабатывает несколько CSV-файлов, создает структурированное хранилище с 32 папками и одним уровнем.

### Полное перемещение файлов из CSV
```bash
python audio_processor.py dataset.csv -fm dataset.csv -o moved_dataset
```
Перемещает все файлы, указанные в dataset.csv, в директорию moved_dataset без конвертации.
