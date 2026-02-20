'''Скрипт для внедрения табличных данных в базу данных'''

import json
import os
import datetime
import re
from typing import List
from langchain_core.documents import Document

# ИМПОРТИРУЕМ НАШ МЕНЕДЖЕР
from db_manager import get_db_connection

JSON_PATH = os.path.join("data/table_parser_files", "stankin_programs.json")

def clean_int(value) -> int:
    """Превращает строку '182 100' или '70' в число 182100. Если мусор - возвращает 0."""
    if isinstance(value, int): return value
    try:
        return int(str(value).replace(" ", ""))
    except (ValueError, TypeError):
        return 0
    
def prettify_exams(raw_text: str) -> str:
    """Делает текст экзаменов читаемым."""
    if not raw_text or raw_text == 'N/A':
        return "Нет данных о вступительных испытаниях"

    mapping = {
        'Р': 'Русский язык', 'М': 'Математика', 'И': 'Информатика',
        'Ф': 'Физика', 'Х': 'Химия', 'X': 'Химия', 'О': 'Обществознание',
        'ИЯ': 'Иностранный язык', 'Б': 'Биология', 'Г': 'География', 'Л': 'Литература'
    }

    text = raw_text
    for short, full in mapping.items():
        text = re.sub(rf'\b{short}\b', full, text)

    text = text.replace("min", "минимум").replace("+", ",").replace("/", " или ")
    return re.sub(r'\s+', ' ', text).strip()


def create_documents(path: str) -> List[Document]:
    if not os.path.exists(path):
        print(f"Файл {path} не найден! Текущая папка: {os.getcwd()}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    print(f"Обработка {len(data)} программ...")

    for prog in data:
        raw_subjects_list = prog.get('Предметы_Список', [])
        subjects_str = ", ".join(raw_subjects_list)
        exams_pretty = prettify_exams(prog.get('Предметы', ''))

        metadata = {
            "source_type": "Таблица",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d"),
            "program_code": prog.get('Код', 'global'),
            "form": prog.get('Форма', 'очная'),
            "level": prog.get('Уровень', 'Бакалавриат'),
            "subjects": subjects_str, 
            "b_places": clean_int(prog.get('Бюджет', 0)),
            "p_rf_places": clean_int(prog.get('Платное_РФ', 0)),
            "p_in_places": clean_int(prog.get('Платное_Иностр', 0)),
            "price_rf": clean_int(prog.get('Стоимость_РФ', 0)),
            "price_in": clean_int(prog.get('Стоимость_Иностр', 0)),
            "score_last": clean_int(prog.get('Балл_2025', 0))
        }

        content = f"""
Направление: {prog['Код']} {prog['Направление']}
Уровень образования: {prog['Уровень']}
Форма обучения: {prog['Форма']}

Экзамены (ЕГЭ): {exams_pretty}

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

        documents.append(Document(page_content=content, metadata=metadata))

    return documents

def main():
    docs = create_documents(JSON_PATH)
    if not docs:
        return

    # 1. ЗАПРАШИВАЕМ БАЗУ У МЕНЕДЖЕРА
    vectorstore = get_db_connection()
    
    # 2. ДОБАВЛЯЕМ ДОКУМЕНТЫ
    print("Добавление таблиц в базу...")
    vectorstore.add_documents(documents=docs)
    
    print(f"УСПЕХ! Загружено {len(docs)} объектов. База обновлена.")

if __name__ == "__main__":
    main()