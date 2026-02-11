import json
import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import datetime

# --- НАСТРОЙКИ ПУТЕЙ ---
# Папка, куда ты сложил JSON-файлы подкастов
PODCASTS_DIR = os.path.join("Data/audio", "jsons")
# Путь к ТЕКУЩЕЙ базе данных (где уже лежат таблицы)
CHROMA_PATH = "Data/chroma_db" 

def create_documents_from_podcasts(directory: str) -> List[Document]:
    """
    Читает JSON-файлы подкастов и превращает их в объекты Document
    с богатым контекстом и метаданными.
    """
    if not os.path.exists(directory):
        print(f"❌ Ошибка: Папка {directory} не найдена!")
        return []

    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    documents = []
    print(f"🎙️ Найдено файлов подкастов: {len(files)}. Начинаем обработку...")

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

                # 2. Проходим по сегментам (смысловым кускам)
                for segment in podcast.get('segments', []):
                    text = segment.get('text', '').strip()
                    if not text: continue

                    # Обработка ключевых слов (превращаем список в строку)
                    keywords_raw = segment.get('keywords', [])
                    if isinstance(keywords_raw, list):
                        keywords_str = ", ".join(keywords_raw)
                    else:
                        keywords_str = str(keywords_raw)

                    # Тип сегмента: summary (якорь) или dialogue (детали)
                    seg_type = segment.get('segment_type', 'dialogue')

                    # --- ФОРМИРОВАНИЕ ТЕКСТА (RICH CONTENT) ---
                    # Мы "вшиваем" контекст прямо в текст, чтобы нейросеть понимала, 
                    # о чем речь, даже если найдет маленький кусочек.
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
                        "source_type": "podcast",      # Маркер источника (ВАЖНО!)
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "program_code": prog_code,     # Ключ для связи с таблицей
                        "speaker": speaker,
                        "role": role,
                        "segment_type": seg_type,      # 'summary' или 'dialogue'
                        "keywords": keywords_str,
                        "url": url
                        
                    }

                    documents.append(Document(page_content=page_content, metadata=metadata))

        except Exception as e:
            print(f"❌ Ошибка при чтении {filename}: {e}")

    return documents

def main():
    # 1. Генерация документов из файлов
    docs = create_documents_from_podcasts(PODCASTS_DIR)
    
    if not docs:
        print("⚠️ Нет документов для загрузки. Проверьте папку Data/podcasts.")
        return

    print(f"📄 Подготовлено {len(docs)} фрагментов (чанков).")

    # 2. Инициализация модели (ОБЯЗАТЕЛЬНО ТА ЖЕ, ЧТО И ДЛЯ ТАБЛИЦ!)
    print("🧠 Загрузка модели эмбеддингов (intfloat/multilingual-e5-large)...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 3. Подключение к существующей базе и добавление данных
    print(f"💾 Подключение к базе '{CHROMA_PATH}'...")
    
    # Внимание: здесь мы НЕ удаляем папку (shutil.rmtree), а просто подключаемся
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    
    print("🚀 Добавление новых документов...")
    vectorstore.add_documents(documents=docs)
    
    print(f"✅ УСПЕХ! В базу добавлено {len(docs)} фрагментов подкастов.")
    print("Теперь база содержит данные и из таблиц, и из подкастов.")

if __name__ == "__main__":
    main()