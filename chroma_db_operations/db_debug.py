'''Скрипт для отладки векторной базы данных. 
Проверяем насколько хорошо работает семантический поиск'''

import json
from db_manager import get_db_connection

def main():
    print("Запуск системы отладки векторной базы...\n")

    # 1. ПОДКЛЮЧЕНИЕ К БАЗЕ
    try:
        vectorstore = get_db_connection()
    except Exception as e:
        print(f"Ошибка подключения к базе: {e}")
        return

    # 2. ТЕСТОВЫЕ ЗАПРОСЫ
    # Задаем вопросы, которых НЕТ в тексте напрямую, чтобы проверить смысловой поиск
    queries = [
        "Где меньше всего физики?",
        "Я хочу разрабатывать танки",
        "Куда поступить чтобы стать Data Science специалистом?",
        "Где разрабатывают роботов?",
        "Какое направление самое интересное?",
        "Сколько стоит обучение на Программиста?",
        "Какой проходной балл на Прикладную информатику в 2025 году?",
    ]

    for q in queries:
        print(f"\n{'='*50}")
        print(f"ВОПРОС: {q}")
        print(f"{'='*50}")
        
        # Ищем 6 самых подходящих документов
        results = vectorstore.similarity_search_with_score(q, k=6)

        if not results:
            print("⚠️ По запросу ничего не найдено.")
            continue

        for i, (doc, score) in enumerate(results):
            # В ChromaDB score — это расстояние (L2). Чем меньше, тем лучше.
            quality = "🟢 ОТЛИЧНО" if score < 0.4 else "🟡 НОРМ" if score < 0.8 else "🔴 ТАК СЕБЕ"
            
            print(f"\nДокумент №{i+1} | Дистанция: {score:.4f} [{quality}]")
            
            # Безопасно выводим метаданные, опираясь на наши новые стандарты
            meta = doc.metadata
            print(f"Код: {meta.get('program_code', 'Не указан')}")
            print(f"Тип источника: {meta.get('source_type', 'Неизвестно')}")
            
            # Если это подкаст, выводим специфичные поля
            if meta.get('source_type') == 'Подкаст':
                print(f"Спикер: {meta.get('speaker', '-')}")
                print(f"Ключевые слова: {meta.get('keywords', '-')}")
            
            # Если это таблица, можно вывести балл или цену
            if meta.get('source_type') == 'Таблица':
                print(f"Цена (РФ): {meta.get('price_rf', '-')} руб.")
                print(f"Проходной балл: {meta.get('score_last', '-')}")
            
            print("-" * 15 + " ТЕКСТ ДОКУМЕНТА " + "-" * 15)
            print(doc.page_content)
            print("-" * 47)

    # --- 3. ПОДКАПОТНЫЕ ДАННЫЕ (ВЕКТОРЫ И ID) ---
    print("\n\n" + "*"*50)
    print("АНАЛИЗ СТРУКТУРЫ БД (МЕТОД .get())")
    print("*"*50)
    
    # Достаем 1 любой документ вместе с его эмбеддингами
    data = vectorstore.get(limit=1, include=['metadatas', 'documents', 'embeddings'])

    if not data or not data.get('ids'):
        print("База пуста!")
        return

    doc_id = data['ids'][0]
    metadata = data['metadatas'][0]
    content = data['documents'][0]
    embedding = data['embeddings'][0]

    print(f"\nID документа в базе: {doc_id}")

    print("\n1. МЕТАДАННЫЕ:")
    print(json.dumps(metadata, indent=4, ensure_ascii=False))

    print("\n2. ТЕКСТ (То, что отправляется в нейросеть для генерации ответа):")
    print("-" * 20)
    print(content)
    print("-" * 20)

    print("\n3. ВЕКТОР:")
    print(f"Всего измерений (размерность): {len(embedding)}")
    print(f"Первые 10 чисел: {embedding[:10]}")
    print("... и так далее.")

    print(f"\n{'='*50}")
    print("Итог: Отладка завершена. Подключение и извлечение данных работают корректно.")

if __name__ == "__main__":
    main()