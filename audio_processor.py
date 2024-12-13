import csv
import hashlib
import multiprocessing
import os
import random
import shutil
import string
import subprocess
import sys
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import argparse


def generate_unique_filename(source_path: str, target_ext: str | None = None) -> str:
    """
    Генерирует уникальное имя файла с помощью хеша

    Args:
        source_path (str): Путь к исходному файлу
        target_ext (str, optional): Расширение для целевого файла. 
                                    Если не указано, используется оригинальное.

    Returns:
        str: Сгенерированное уникальное имя файла с расширением
    """
    if not target_ext:
        target_ext = os.path.splitext(source_path)[1]
    
    target_ext = target_ext.lstrip('.')
    
    code = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(16))
    hash_object = hashlib.md5((code+source_path).encode())
    hex_digest = hash_object.hexdigest()
    return f"{hex_digest}.{target_ext}"


def create_directory_structure(base_path: str, depth: int = 16) -> None:
    """
    Создает гибкую структуру директорий

    Args:
        base_path (str): Базовый путь для создания структуры
        depth (int, optional): Глубина структуры директорий. 
                               По умолчанию 16, возможные значения 16, 32 и т.д.

    Создает вложенные директории с использованием шестнадцатеричных имен, 
    обеспечивая эффективное распределение файлов по подпапкам.
    """
    # Генерируем список имен папок в зависимости от глубины
    folder_names = [format(i, f'0{len(str(depth-1))}x') for i in range(depth)]
    
    for i in folder_names:
        for j in folder_names:
            path = os.path.join(base_path, i, j)
            os.makedirs(path, exist_ok=True)


def get_target_path(file_index: int, base_path: str, filename: str, depth: int = 16) -> str:
    """
    Определяет путь назначения для файла на основе его индекса
    с учетом переменной глубины структуры директорий

    Args:
        file_index (int): Индекс файла для распределения
        base_path (str): Базовый путь для сохранения
        filename (str): Имя файла
        depth (int, optional): Глубина структуры директорий. По умолчанию 16.

    Returns:
        str: Полный путь к файлу с учетом распределения по подпапкам
    """
    folder_names = [format(i, f'0{len(str(depth-1))}x') for i in range(depth)]
    
    dir_index = (file_index // (depth * depth)) % depth
    subdir_index = (file_index // depth) % depth
    
    return os.path.join(
        base_path, 
        folder_names[dir_index], 
        folder_names[subdir_index], 
        filename
    )


def convert_audio_ffmpeg(source_path: str, target_path: str, ffmpeg_params: list[str] | None = None) -> bool:
    """
    Конвертирует или копирует файл через ffmpeg с расширенными параметрами

    Args:
        source_path (str): Путь к исходному файлу
        target_path (str): Путь для сохранения конвертированного файла
        ffmpeg_params (list, optional): Дополнительные параметры для ffmpeg

    Returns:
        bool: True, если конвертация прошла успешно, иначе False
    """
    try:        
        # Создаем директорию для целевого файла, если она не существует
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # Базовая команда ffmpeg
        command = [
            'ffmpeg',
            '-i', source_path,
            '-loglevel', 'panic',  # Отключение вывода в терминал
            '-y'
        ]
        
        # Добавляем пользовательские параметры ffmpeg, если они указаны
        if ffmpeg_params:
            command.extend(ffmpeg_params)
        
        # Добавляем целевой путь в конец команды
        command.append(target_path)
        
        # Перенаправляем вывод ffmpeg в никуда для ускорения
        with open(os.devnull, 'wb') as devnull:
            subprocess.check_call(command, stdout=devnull, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(f"Ошибка при обработке {source_path}: {str(e)}")
        return False


def process_batch(args: tuple) -> list[list[str]]:
    """
    Обработка пакета файлов с конвертацией или перемещением

    Args:
        args (tuple): Кортеж с параметрами обработки:
            - batch_files (list): Список файлов для обработки
            - output_base_path (str): Базовый путь вывода
            - convert_formats (list): Форматы для конвертации
            - move_formats (list): Форматы для перемещения
            - target_ext (str): Целевое расширение
            - full_move_csvs (list): CSV-файлы для полного перемещения
            - ffmpeg_params (list): Параметры ffmpeg
            - depth (int): Глубина структуры директорий

    Returns:
        list: Список обработанных файлов с новыми путями, текстом и длительностью
    """
    (batch_files, output_base_path, convert_formats, move_formats, target_ext, 
     full_move_csvs, ffmpeg_params, depth) = args
    
    results = []
    for index, row in batch_files:
        source_path = row[0]
        text = row[1]
        duration = round(float(row[2]), 2)
        
        # Определяем расширение исходного файла
        file_ext = os.path.splitext(source_path)[1].lower()
        
        # Генерируем уникальное имя файла с сохранением расширения
        unique_filename = generate_unique_filename(source_path, target_ext)
        target_path = get_target_path(index, output_base_path, unique_filename, depth)
        
        success = False
        # Решаем, что делать с файлом
        if any(os.path.basename(csv_file) in full_move_csvs for csv_file in full_move_csvs):
            # Если файлы из CSV, которые нужно все переместить
            success = convert_audio_ffmpeg(source_path, target_path)
        elif file_ext in move_formats:
            # Если файл с таким форматом нужно переместить
            success = convert_audio_ffmpeg(source_path, target_path)
        elif file_ext in convert_formats:
            # Если файл с таким форматом нужно конвертировать
            success = convert_audio_ffmpeg(source_path, target_path, ffmpeg_params)
        
        if success:
            results.append([target_path, text, duration])
    return results


def chunks(lst: list, n: int) -> list:
    """
    Разделяет список на чанки заданного размера

    Args:
        lst (list): Входной список для разделения
        n (int): Размер каждого чанка

    Yields:
        list: Чанки входного списка
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_csv_files(csv_files: list[str], full_move_csvs: list[str]) -> tuple[list, dict, list, list]:
    """
    Читает несколько CSV файлов, удаляет дубликаты и возвращает обработанные строки
    
    Args:
        csv_files (list): Список путей к CSV-файлам
        full_move_csvs (list): Список CSV-файлов для полного перемещения
    
    Returns:
        tuple: Кортеж со следующими элементами:
            - rows (list): Обработанные строки без дубликатов
            - file_counts (dict): Статистика по файлам
            - full_move_sources (list): Источники для полного перемещения
            - formats_found (list): Найденные форматы файлов
    """
    rows = []
    file_counts = {}
    full_move_sources = []
    formats_found = set()
    total_duplicates = 0
    
    print("Начало обработки CSV-файлов...")
    print("Удаление дубликатов...")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, delimiter="|")
            total_entries_in_file = len(df)

            # Удаляем дубликаты, сохраняя первое вхождение по второму столбцу
            df_no_duplicates = df.drop_duplicates(subset=[df.columns[1]], keep='first')

            duplicates_in_file = total_entries_in_file - len(df_no_duplicates)
            total_duplicates += duplicates_in_file

            print(f" - Файл {os.path.basename(csv_file)}:")
            print(f"   Всего записей: {total_entries_in_file}")
            print(f"   Дубликатов: {duplicates_in_file}")
            print(f"   Уникальных записей: {total_entries_in_file - duplicates_in_file}")

            # Преобразуем обратно в список списков для совместимости с оригинальным кодом
            for row in df_no_duplicates.values.tolist():
                rows.append((len(rows), row))

                file_ext = os.path.splitext(row[0])[1].lower()
                formats_found.add(file_ext)
                file_counts[file_ext] = file_counts.get(file_ext, 0) + 1

                if os.path.basename(csv_file) in full_move_csvs:
                    full_move_sources.append(os.path.basename(row[0]))
        except FileNotFoundError:
            print(f"Ошибка: Файл {csv_file} не найден")
        except pd.errors.EmptyDataError:
            print(f"Ошибка: Файл {csv_file} пустой")
        except Exception as e:
            print(f"Произошла ошибка при обработке файла {csv_file}: {str(e)}")

    # Итоговая статистика по дубликатам
    print(f"\nВсего дубликатов удалено: {total_duplicates}")

    return rows, file_counts, full_move_sources, list(formats_found)


def main():
    """
    Основная функция для обработки аудиофайлов с поддержкой конвертации, 
    перемещения и создания структурированного хранилища

    Обрабатывает аргументы командной строки, читает входные CSV-файлы, 
    создает структуру директорий, обрабатывает файлы пакетами с использованием 
    многопроцессорной обработки и сохраняет результаты в директорию и новый CSV-файл.
    """
    # Парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Конвертер аудиофайлов с расширенными настройками')
    parser.add_argument('csv_files', nargs='+', help='Путь к CSV файлам')
    parser.add_argument('-o', '--output', default='dataset', help='Базовая директория для вывода')
    parser.add_argument('-oc', '--output-csv', default='metadata.csv', help='Имя выходного CSV файла')
    parser.add_argument('-hd', '--headers', nargs='*', default=['audio_path', 'text', 'duration'], help='Заголовки для выходного CSV')
    parser.add_argument('-d', '--depth', type=int, default=16, help='Глубина структуры директорий (16х16, 32х32 и т.д.)')
    parser.add_argument('-c', '--convert', nargs='*', help='Форматы файлов для конвертации')
    parser.add_argument('-m', '--move', nargs='*', default=[], help='Форматы файлов для перемещения')
    parser.add_argument('-t', '--target-ext', nargs='*', default=None, help='Целевое расширение файла для конвертации')
    parser.add_argument('-fm', '--full-move', nargs='*', default=[], help='CSV-файлы для полного перемещения')
    parser.add_argument('--ffmpeg', nargs='*', default=['16000', '64k', 'libmp3lame'], help='Параметры ffmpeg: частота, битрейт, кодек')
    parser.add_argument('-bs', '--batch-size', type=int, default=50, help='Размер пакета файлов')
    
    args = parser.parse_args()
    
    #Запись параметров для ffmpeg
    if args.ffmpeg:
        ffmpeg_args = []
        if len(args.ffmpeg) >= 1:
            ffmpeg_args.append('-ar')
            ffmpeg_args.append(args.ffmpeg[0])
        if len(args.ffmpeg) >= 2:
            ffmpeg_args.append('-b:a')
            ffmpeg_args.append(args.ffmpeg[1])
        if len(args.ffmpeg) >= 3:
            ffmpeg_args.append('-c:a')
            ffmpeg_args.append(args.ffmpeg[2])
        if len(args.ffmpeg) > 3:
            print("Предупреждение: Указано слишком много параметров. Лишние параметры будут проигнорированы.")
        args.ffmpeg = ffmpeg_args

    print("Создание структуры директорий...")
    create_directory_structure(args.output, args.depth)
    
    print("Чтение входных CSV файлов...")
    rows, file_counts, full_move_sources, found_formats = read_csv_files(
        args.csv_files, args.full_move
    )
    
    # Если форматы не указаны, используем найденные в CSV для конвертации
    if not args.convert:
        args.convert = found_formats
    
    total_files = len(rows)
    print(f"Найдено {total_files} файлов:")
    for ext, count in file_counts.items():
        print(f" - {count} {ext.upper()} файлов")
    
    print("Настройки:")
    print(f" - Конвертация форматов: {args.convert}")
    print(f" - Перемещение форматов: {args.move}")
    print(f" - Полное перемещение из CSV: {args.full_move}")
    print(f" - Параметры ffmpeg: {args.ffmpeg}")
    print(f" - Глубина структуры директорий: {args.depth}")
    
    # Разбиваем файлы на батчи
    batches = list(chunks(rows, args.batch_size))
    process_args = [(
        batch, 
        args.output, 
        args.convert, 
        args.move,
        args.target_ext, 
        args.full_move, 
        args.ffmpeg, 
        args.depth
    ) for batch in batches]
    
    # Оптимальное количество процессов
    num_processes = multiprocessing.cpu_count()
    
    # Обработка файлов с использованием пула процессов
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_batch, args) for args in process_args]
        
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Обработка батчей",
            unit="батчей"
        ):
            batch_results = future.result()
            results.extend(batch_results)
    
    # Сохраняем результаты в новый CSV
    output_csv = os.path.join(args.output, args.output_csv)
    print("Сохранение результатов...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f, delimiter='|')
        csv_writer.writerow(args.headers)
        csv_writer.writerows(results)
    
    print(f"Обработка завершена. Обработано {len(results)} из {total_files} файлов.")
    print(f"Результаты сохранены в {output_csv}")


if __name__ == "__main__":
    main()
