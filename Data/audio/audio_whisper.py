import whisper
import os
import logging
from typing import List

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- КОНСТАНТЫ ---
# Автоматически определяем текущую рабочую папку
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Используем модель 'medium' для лучшей точности на русском языке.
MODEL_SIZE = "medium" 
# Расширение файлов, которые мы ищем
FILE_EXTENSION = ".mp3"

def find_audio_files(directory: str, extension: str) -> List[str]:
    """
    Находит все файлы с заданным расширением в указанной директории.
    """
    logging.info(f"Сканирование папки {directory} на наличие файлов *{extension}...")
    
    # Используем list comprehension для быстрого поиска и формирования полных путей
    audio_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(extension)
    ]
    
    if not audio_paths:
        logging.warning(f"Файлы *{extension} не найдены в папке: {directory}")
    else:
        logging.info(f"Найдено {len(audio_paths)} файлов для транскрипции.")
    
    return audio_paths

def transcribe_audio_files(audio_paths: List[str], model_size: str) -> dict:
    """
    Использует модель Whisper для транскрибирования списка аудиофайлов.
    """
    
    logging.info(f"Загрузка модели Whisper: {model_size}.")
    try:
        # Модель загружается только один раз
        model = whisper.load_model(model_size) 
    except Exception as e:
        logging.error(f"Не удалось загрузить модель Whisper: {e}. Проверьте установку зависимостей и FFmpeg.")
        return {}

    results = {}
    
    # 2. Транскрипция каждого файла
    for path in audio_paths:
        filename = os.path.basename(path)
        logging.info(f"\n--- Начало транскрипции: {filename} ---")
        
        try:
            result = model.transcribe(
                audio=path, 
                language="ru",
                verbose=False
            )
            
            results[filename] = result["text"]
            logging.info(f"УСПЕХ: Транскрипция {filename} завершена.")
            
        except Exception as e:
            logging.error(f"ОШИБКА при транскрипции {filename}: {e}")
            results[filename] = f"ERROR: {e}"

    return results

def save_results(transcription_results: dict, base_dir: str):
    """
    Сохраняет результаты транскрипции в отдельные TXT-файлы.
    """
    for filename, text in transcription_results.items():
        if not text.startswith("ERROR"):
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(base_dir, output_filename)
            
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                print(f"✅ УСПЕШНО: Транскрипция [{filename}] сохранена в: {output_path}")
                
            except Exception as e:
                print(f"❌ ОШИБКА: Не удалось сохранить файл {output_filename}: {e}")
        else:
            print(f"❌ ОШИБКА: Файл [{filename}] не был транскрибирован.")


# --- ОСНОВНАЯ ЛОГИКА ЗАПУСКА ---

if __name__ == "__main__":
    
    # 1. Поиск всех MP3-файлов в текущей папке
    podcast_files = find_audio_files(BASE_DIR, FILE_EXTENSION)
    
    if podcast_files:
        # 2. Запуск транскрипции
        transcription_results = transcribe_audio_files(podcast_files, MODEL_SIZE)
        
        # 3. Сохранение результатов
        print("\n" + "="*50)
        print("СОХРАНЕНИЕ ВСЕХ ТРАНСКРИПЦИЙ")
        print("="*50)
        save_results(transcription_results, BASE_DIR)
        
    print("\nПакетная транскрипция завершена.")