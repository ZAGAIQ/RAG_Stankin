'''Скрипт для внедрения подкастов в базу данных'''

import json
import os
import datetime
from typing import List
from langchain_core.documents import Document

from db_manager import get_db_connection

PODCASTS_DIR = os.path.join("data/audio", "jsons")

def create_documents_from_podcasts(directory: str) -> List[Document]:
    """
    Читает JSON-файлы подкастов и превращает их в объекты Document
    с богатым контекстом и метаданными.
    """
    if not os.path.exists(directory):
        print(f"Ошибка: Папка {directory} не найдена!")
        return []

    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    documents = []
    print(f"Найдено файлов подкастов: {len(files)}. Начинаем обработку...")

    for filename in files:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Если в файле один объект, превращаем в список для унификации
            if isinstance(data, dict):
                data = [data]

            for podcast in data:
                # 1. Извлекаем глобальные данные подкаста
                prog_code = podcast.get('program_code', 'global')
                prog_name = podcast.get('program_name', 'Неизвестно')
                speaker = podcast.get('speaker', 'Эксперт')
                role = podcast.get('role', 'Сотрудник вуза')
                url = podcast.get('url', '')

                # 2. Проходим по сегментам
                for segment in podcast.get('segments', []):
                    text = segment.get('text', '').strip()
                    if not text: continue

                    # Обработка ключевых слов
                    keywords_raw = segment.get('keywords', [])
                    if isinstance(keywords_raw, list):
                        keywords_str = ", ".join(keywords_raw)
                    else:
                        keywords_str = str(keywords_raw)

                    # Тип сегмента: summary (якорь) или dialogue (детали)
                    seg_type = segment.get('segment_type', 'dialogue')

                    # --- ФОРМИРОВАНИЕ ТЕКСТА (RICH CONTENT) ---
                    page_content = f"""
Источник: Подкаст о направлении {prog_code} "{prog_name}".
Спикер: {speaker} ({role}).
Тип информации: {"Обзор направления" if seg_type == 'summary' else "Детали и ответы на вопросы"}
Ключевые темы: {keywords_str}
Текст:
{text}
""".strip()

                    # --- МЕТАДАННЫЕ (ДЛЯ ФИЛЬТРОВ) ---
                    metadata = {
                        "source_type": "Подкаст", 
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "program_code": prog_code,
                        "speaker": speaker,
                        "role": role,
                        "segment_type": seg_type,
                        "keywords": keywords_str,
                        "url": url
                    }

                    documents.append(Document(page_content=page_content, metadata=metadata))

        except Exception as e:
            print(f"Ошибка при чтении {filename}: {e}")

    return documents


def main():
    # 1. Генерация документов из файлов
    docs = create_documents_from_podcasts(PODCASTS_DIR)
    
    if not docs:
        print("Нет документов для загрузки. Проверьте папку Data/audio/jsons.")
        return

    print(f"Подготовлено {len(docs)} фрагментов (чанков).")

    # 2. ПОДКЛЮЧЕНИЕ К БАЗЕ ЧЕРЕЗ МЕНЕДЖЕР
    vectorstore = get_db_connection()
    
    # 3. ДОБАВЛЕНИЕ ДОКУМЕНТОВ В БАЗУ
    print("Добавление фрагментов подкастов в базу...")
    vectorstore.add_documents(documents=docs)
    
    print(f"УСПЕХ! В базу добавлено {len(docs)} фрагментов подкастов.")

if __name__ == "__main__":
    main()