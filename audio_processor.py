import csv
import hashlib
import multiprocessing
import os
import logging
import random
import shutil
import string
import subprocess
import sys
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import argparse
import threading
import queue


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


def create_directory_structure(base_path: str, levels: Union[int, List[int]]) -> None:
    """
    Создает гибкую многоуровневую структуру директорий

    Args:
        base_path (str): Базовый путь для создания структуры
        levels (Union[int, List[int]]): Конфигурация уровней структуры директорий.
            Может быть:
            - Одним целым числом (например, 16) для равномерной структуры
            - Списком чисел для настройки каждого уровня (например, [4, 7, 10, 23])
    """
    def generate_folder_names(level_depth: int) -> List[str]:
        """Генерирует список имен папок для указанного уровня."""
        return [format(i, f'0{len(str(level_depth-1))}x') for i in range(level_depth)]

    # Нормализация входных данных
    if isinstance(levels, int):
        levels = [levels]

    # Проверка корректности входных данных
    if not levels or not all(isinstance(x, int) and x > 0 for x in levels):
        raise ValueError("Levels должен быть целым числом или списком положительных целых чисел")

    # Генерация многоуровневой структуры
    def create_nested_dirs(current_path: str, current_level: int):
        if current_level >= len(levels):
            return

        folder_names = generate_folder_names(levels[current_level])
        
        for folder in folder_names:
            next_path = os.path.join(current_path, folder)
            os.makedirs(next_path, exist_ok=True)
            create_nested_dirs(next_path, current_level + 1)

    create_nested_dirs(base_path, 0)


def get_target_path(file_index: int, base_path: str, filename: str, levels: Union[int, List[int]]) -> str:
    """
    Определяет путь назначения для файла на основе его индекса
    с учетом переменной конфигурации уровней директорий

    Args:
        file_index (int): Индекс файла для распределения
        base_path (str): Базовый путь для сохранения
        filename (str): Имя файла
        levels (Union[int, List[int]], optional): Конфигурация уровней.
            Может быть одним числом или списком чисел.

    Returns:
        str: Полный путь к файлу с учетом распределения по подпапкам
    """
    def generate_folder_names(level_depth: int) -> List[str]:
        """Генерирует список имен папок для указанного уровня."""
        return [format(i, f'0{len(str(level_depth-1))}x') for i in range(level_depth)]

    # Нормализация входных данных
    if isinstance(levels, int):
        levels = [levels]

    if not levels or not all(isinstance(x, int) and x > 0 for x in levels):
        raise ValueError("Levels должен быть целым числом или списком положительных целых чисел")

    # Вычисление индексов для каждого уровня
    path_components = []
    remaining_index = file_index

    for level_depth in levels:
        current_level_index = remaining_index % level_depth
        remaining_index //= level_depth
        folder_names = generate_folder_names(level_depth)
        path_components.append(folder_names[current_level_index])

    return os.path.join(base_path, *path_components, filename)


def convert_audio_ffmpeg(source_path: str, target_path: str, ffmpeg_params: list[str] | None = None, 
                         switch_copy_move: bool = True) -> bool:
    """
    Конвертирует или копирует файл через ffmpeg с расширенными параметрами

    Args:
        source_path (str): Путь к исходному файлу
        target_path (str): Путь для сохранения конвертированного файла
        ffmpeg_params (list, optional): Дополнительные параметры для ffmpeg
        switch_copy_move (bool, optional): Переключение между копированием и перемещением

    Returns:
        bool: True, если конвертация или перемещение/копирование прошло успешно, иначе False
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
        if ffmpeg_params:
            # Добавляем пользовательские параметры ffmpeg
            command.extend(ffmpeg_params)
            # Добавляем целевой путь в конец команды
            command.append(target_path)
            
            if switch_copy_move:
                # Перенаправляем вывод ffmpeg в никуда для ускорения
                with open(os.devnull, 'wb') as devnull:
                    subprocess.check_call(command, stdout=devnull, stderr=subprocess.STDOUT)
                os.remove(source_path)
            else:
                with open(os.devnull, 'wb') as devnull:
                    subprocess.check_call(command, stdout=devnull, stderr=subprocess.STDOUT)
        else:
            if switch_copy_move:
                shutil.move(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
        
        return True
    except Exception as e:
        logging.error(f"Ошибка при обработке {source_path}: {str(e)}")
        return False


def process_batch(args: tuple) -> list[list[str]]:
    """
    Обработка пакета файлов с конвертацией или перемещением

    Args:
        args (tuple): Кортеж с параметрами обработки, включая новые параметры

    Returns:
        list: Список обработанных файлов с новыми путями, текстом и длительностью
    """
    (batch_files, output_base_path, convert_formats, move_formats, target_ext, 
     full_move_csvs, ffmpeg_params, levels, switch_copy_move) = args

    results = []
    for index, row in batch_files:
        source_path = row[0]
        
        # Определяем расширение исходного файла
        file_ext = os.path.splitext(source_path)[1].lower()
        
        # Генерируем уникальное имя файла с сохранением расширения
        unique_filename = generate_unique_filename(source_path, target_ext)
        target_path = get_target_path(index, output_base_path, unique_filename, levels)
        
        success = False
        # Решаем, что делать с файлом
        if source_path in full_move_csvs:
            # Если файлы из CSV, которые нужно все переместить
            success = convert_audio_ffmpeg(source_path, target_path, None, switch_copy_move)
        elif file_ext in move_formats:
            # Если файл с таким форматом нужно переместить
            success = convert_audio_ffmpeg(source_path, target_path, None, switch_copy_move)
        elif file_ext in convert_formats:
            # Если файл с таким форматом нужно конвертировать
            success = convert_audio_ffmpeg(source_path, target_path, ffmpeg_params, switch_copy_move)
        
        if success:
            results.append(row)
    return results


def csv_save_worker(output_csv: str, result_queue: queue.Queue, headers: list, delimiter: str):
    """
    Рабочий поток для сохранения CSV-файла в реальном времени
    
    Args:
        output_csv (str): Путь к выходному CSV-файлу
        result_queue (queue.Queue): Очередь результатов для сохранения
        headers (list): Заголовки CSV-файла
    """
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f, delimiter=delimiter)
        csv_writer.writerow(headers)
        
        while True:
            batch = result_queue.get()
            if batch is None:
                break
            
            csv_writer.writerows(batch)
            f.flush()  # Принудительная запись на диск


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


def read_csv_files(csv_files: list[str], full_move_csvs: list[str], duplicate_index: int, delimiter: str) -> tuple[list, dict, list, list]:
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
    
    for csv_file in csv_files:
        try:
            if duplicate_index:
                df = pd.read_csv(csv_file, delimiter=delimiter)
                total_entries_in_file = len(df)

                # Удаляем дубликаты, сохраняя первое вхождение по столбцу с указаным индексом
                df_no_duplicates = df.drop_duplicates(subset=[df.columns[duplicate_index]], keep='first')

                duplicates_in_file = total_entries_in_file - len(df_no_duplicates)
                total_duplicates += duplicates_in_file

                print(f" - Файл {os.path.basename(csv_file)}:")
                print(f"   Всего записей: {total_entries_in_file}")
                print(f"   Дубликатов: {duplicates_in_file}")
                print(f"   Уникальных записей: {total_entries_in_file - duplicates_in_file}")

                # Преобразуем обратно в список списков для совместимости с оригинальным кодом
                for row in df_no_duplicates.values.tolist():
                    rows.append((len(rows), row))

                    if os.path.basename(csv_file) in full_move_csvs:
                        full_move_sources.append(row[0])

                    file_ext = os.path.splitext(row[0])[1].lower()
                    formats_found.add(file_ext)
                    file_counts[file_ext] = file_counts.get(file_ext, 0) + 1
            else:
                df = pd.read_csv(csv_file, delimiter=delimiter)
                total_entries_in_file = len(df)

                print(f" - Файл {os.path.basename(csv_file)}:")
                print(f"   Всего записей: {total_entries_in_file}")

                # Преобразуем обратно в список списков для совместимости с оригинальным кодом
                for row in df.values.tolist():
                    rows.append((len(rows), row))

                    if os.path.basename(csv_file) in full_move_csvs:
                        full_move_sources.append(row[0])
                    
                    file_ext = os.path.splitext(row[0])[1].lower()
                    formats_found.add(file_ext)
                    file_counts[file_ext] = file_counts.get(file_ext, 0) + 1
        except FileNotFoundError:
            logging.error(f"Ошибка: Файл {csv_file} не найден")
        except pd.errors.EmptyDataError:
            logging.error(f"Ошибка: Файл {csv_file} пустой")
        except Exception as e:
            logging.error(f"Произошла ошибка при обработке файла {csv_file}: {str(e)}")

    # Итоговая статистика по дубликатам
    if duplicate_index:
        print(f"Всего дубликатов удалено: {total_duplicates}")

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
    parser.add_argument('-o', '--output', default='audio_directory', help='Базовая директория для вывода')
    parser.add_argument('-oc', '--output-csv', default='metadata.csv', help='Имя выходного CSV файла')
    parser.add_argument('-de', '--delimiter', type=str, default="|", help='Разделитель в CSV-файлах')
    parser.add_argument('-hd', '--headers', nargs='*', default=['audio_path', 'text', 'duration'], help='Заголовки для выходного CSV')
    parser.add_argument('-l', '--levels', nargs='*', type=int, default=[16], help='Конфигурация уровней структуры директорий')
    parser.add_argument('-c', '--convert', nargs='*', default=None, help='Форматы файлов для конвертации')
    parser.add_argument('-m', '--move', nargs='*', default=[], help='Форматы файлов для перемещения')
    parser.add_argument('-t', '--target-ext', default='.mp3', help='Целевое расширение файла для конвертации')
    parser.add_argument('-fm', '--full-move', nargs='*', default=[], help='CSV-файлы для полного перемещения')
    parser.add_argument('--ffmpeg', type=str, default='-ar 16000 -b:a 64k -c:a libmp3lame', help='Параметры ffmpeg: частота, битрейт, кодек и т.д.')
    parser.add_argument('-bs', '--batch-size', type=int, default=12, help='Размер пакета файлов')
    parser.add_argument('--cpu', type=int, default=multiprocessing.cpu_count(), help='Одновременное количество процессов')
    parser.add_argument('-scm', '--switch-copy-move', type=bool, default=False, help='Изменить копирование на перемещение')
    parser.add_argument('-csr', '--csv-save-realtime', type=bool, default=True, help='Сохраняет CSV-файл во время всего выполнения кода, а не только в конце')
    parser.add_argument('-di', '--duplicate-index', type=int, default=None, help='Указать какой столбец проверить на дубликаты, если не указывать проверки не будет')
    
    args = parser.parse_args()
    args.ffmpeg = args.ffmpeg.split(' ')
    
    #Конфигурация для логирования ошибок
    logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Создание структуры директорий...")
    create_directory_structure(args.output, args.levels)
    
    print("Чтение входных CSV файлов...")
    rows, file_counts, full_move_sources, found_formats = read_csv_files(
        args.csv_files, args.full_move, args.duplicate_index, args.delimiter
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
    print(f' - Разделитель в CSV-файлах: "{args.delimiter}"')
    print(f" - Конфигурация уровней структуры директорий: {args.levels}")
    print(f" - Удаление дубликатов: {'Включено' if args.duplicate_index else 'Выключено'}")
    print(f" - Копирование/перемещение: {'Перемещение' if args.switch_copy_move else 'Копирование'}")
    print(f" - Сохранение CSV в реальном времени: {'Включено' if args.csv_save_realtime else 'Выключено'}")
    
    # Разбиваем файлы на батчи
    batches = list(chunks(rows, args.batch_size))
    process_args = [(
        batch, 
        args.output, 
        args.convert, 
        args.move,
        args.target_ext, 
        full_move_sources, 
        args.ffmpeg, 
        args.levels,
        args.switch_copy_move
    ) for batch in batches]
    
    # Оптимальное количество процессов
    num_processes = args.cpu
    
    # Подготовка для сохранения CSV в реальном времени
    output_csv = os.path.join(args.output, args.output_csv)
    result_queue = queue.Queue()
    
    if args.csv_save_realtime:
        csv_thread = threading.Thread(
            target=csv_save_worker, 
            args=(output_csv, result_queue, args.headers, args.delimiter)
        )
        csv_thread.start()
    
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
            
            if args.csv_save_realtime:
                result_queue.put(batch_results)
    
    # Завершение потока сохранения CSV
    if args.csv_save_realtime:
        result_queue.put(None)  # Сигнал завершения
        csv_thread.join()
    
    # Если CSV не сохранялся в реальном времени, сохраним полностью в конце
    if not args.csv_save_realtime:
        print("Сохранение результатов...")
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f, delimiter=args.delimiter)
            csv_writer.writerow(args.headers)
            csv_writer.writerows(results)
    with open("error.log", "r") as error_file:
        error_file_read = error_file.read()
        if not error_file_read:
            os.remove("error.log")
    
    print(f"Обработка завершена. Обработано {len(results)} из {total_files} файлов.")
    print(f"Результаты сохранены в {output_csv}")


if __name__ == "__main__":
    main()
    
