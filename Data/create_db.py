import json
import os
import datetime
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- НАСТРОЙКИ ---
# Проверь, что имя файла точное. В твоем коде было "Data/table_parser_files", я оставил как у тебя.
JSON_PATH = os.path.join("Data/table_parser_files", "stankin_programs.json")
CHROMA_PATH = "Data/chroma_db"

def clean_int(value) -> int:
    """Превращает строку '182 100' или '70' в число 182100. Если мусор - возвращает 0."""
    if isinstance(value, int): return value
    try:
        return int(str(value).replace(" ", ""))
    except (ValueError, TypeError):
        return 0

def create_documents(path: str) -> List[Document]:
    if not os.path.exists(path):
        print(f"❌ Файл {path} не найден!")
        # Если не найдет, выведем текущую директорию, чтобы ты понял, где скрипт ищет файл
        print(f"Текущая папка: {os.getcwd()}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    print(f"🔄 Обработка {len(data)} программ...")

    for prog in data:
        # 1. ПОДГОТОВКА ДАННЫХ
        # Берем русские значения напрямую из JSON
        
        # ЛЕЧИМ ОШИБКУ "ValueError... list":
        # Превращаем список ["Информатика", "Математика"] в строку "Информатика, Математика"
        # Это позволит ChromaDB "проглотить" данные, а фильтр 'contains' все равно сработает.
        raw_subjects_list = prog.get('Предметы_Список', [])
        subjects_str = ", ".join(raw_subjects_list)

        # 2. СБОРКА МЕТАДАННЫХ (Ключи English, Значения Russian)
        metadata = {
            "source_type": "table",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d"),
            
            # Строковые поля (оставляем как в JSON)
            "program_code": prog.get('Код', 'global'),
            "form": prog.get('Форма', 'очная'),      # Будет "очная"
            "level": prog.get('Уровень', 'Бакалавриат'), # Будет "Бакалавриат"
            
            # Поле с предметами (СТРОКА, не список!)
            "subjects": subjects_str, 
            
            # Числовые поля (int)
            "b_places": clean_int(prog.get('Бюджет', 0)),
            "p_rf_places": clean_int(prog.get('Платное_РФ', 0)),
            "p_in_places": clean_int(prog.get('Платное_Иностр', 0)),
            
            "price_rf": clean_int(prog.get('Стоимость_РФ', 0)),
            "price_in": clean_int(prog.get('Стоимость_Иностр', 0)),
            
            "score_last": clean_int(prog.get('Балл_2025', 0))
        }

        # 3. СБОРКА ТЕКСТА (PAGE CONTENT)
        # То, что читает LLM. Красивый русский текст.
        content = f"""
Направление: {prog['Код']} {prog['Направление']}
Уровень образования: {prog['Уровень']}
Форма обучения: {prog['Форма']}

Вступительные экзамены (ЕГЭ): {prog.get('Предметы', 'Нет данных')}

Количество мест (на 2026 год):
- Бюджетных мест: {metadata['b_places']}
- Платных мест (для РФ): {metadata['p_rf_places']}
- Платных мест (для иностранцев): {metadata['p_in_places']}

Стоимость обучения (за семестр):
- Для граждан РФ: {metadata['price_rf']} руб.
- Для иностранных граждан: {metadata['price_in']} руб.

Проходные баллы прошлых лет (Бюджет):
2025 год: {prog.get('Балл_2025', '-')}
2024 год: {prog.get('Балл_2024', '-')}
2023 год: {prog.get('Балл_2023', '-')}
2022 год: {prog.get('Балл_2022', '-')}
2021 год: {prog.get('Балл_2021', '-')}
""".strip()

        # Создаем документ
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    return documents

def main():
    # 1. Генерация документов
    docs = create_documents(JSON_PATH)
    if not docs:
        return

    # Выведем пример для проверки
    print("\n--- ПРИМЕР МЕТАДАННЫХ (№1) ---")
    print(json.dumps(docs[0].metadata, indent=4, ensure_ascii=False))
    print("------------------------------\n")

    # 2. Инициализация модели
    print("🧠 Загрузка модели эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 3. Сохранение в базу
    print(f"💾 Создание базы данных в '{CHROMA_PATH}'...")
    
    # Удаляем старую базу, чтобы не было конфликтов
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ УСПЕХ! Векторная база создана. Загружено {len(docs)} объектов.")

if __name__ == "__main__":
    main()