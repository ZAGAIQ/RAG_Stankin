'''Код для отладки векторной базы данных ChromaDB'''


import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Путь должен быть ТОЧНО такой же, как в create_db.py
CHROMA_PATH = "Data/chroma_db"

def main():
    # 1. ПРОВЕРКА ПУТИ
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ ОШИБКА: Папка {CHROMA_PATH} не найдена!")
        print("Сначала запусти create_db.py")
        return

    print("🧠 Загружаем модель (секундочку)...")
    # Используем ту же модель, что и при создании!
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 2. ПОДКЛЮЧЕНИЕ К БАЗЕ
    print(f"📂 Подключаемся к базе в '{CHROMA_PATH}'...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

    # 3. ТЕСТОВЫЕ ЗАПРОСЫ
    # Давай зададим вопрос, которого НЕТ в тексте напрямую, чтобы проверить "умный поиск"
    queries = [
        "Где я буду разрабатывать программные комплексы?",
        "Какая специальность связана с управлением?",
        "Куда поступить с химией?"
    ]

    for q in queries:
        print(f"\n{'='*40}")
        print(f"❓ ВОПРОС: {q}")
        print(f"{'='*40}")
        
        # Ищем 2 самых подходящих документа
        results = vectorstore.similarity_search_with_score(q, k=3)

        for i, (doc, score) in enumerate(results):
            quality = "🟢 ОТЛИЧНО" if score < 0.4 else "🟡 НОРМ" if score < 0.8 else "🔴 ТАК СЕБЕ"
            
            print(f"\n📄 Документ №{i+1} | Оценка (Distance): {score:.4f} [{quality}]")
            print(f"📌 Код: {doc.metadata.get('program_code')}")
            print(f"📝 Текст: {doc.page_content[:100].replace(chr(10), ' ')}...") # Убираем переносы строк для красоты
            print("-" * 30)

    # --- САМОЕ ИНТЕРЕСНОЕ: Метод .get() ---
    # Он позволяет достать данные по ID или просто первые попавшиеся (limit)
    # include=['metadatas', 'documents', 'embeddings'] говорит, ЧТО именно достать.
    data = vectorstore.get(limit=1, include=['metadatas', 'documents', 'embeddings'])

    # Chroma возвращает списки, так как мы могли запросить 10 документов
    if not data['ids']:
        print("База пуста!")
        return

    # Берем первый элемент из списков
    doc_id = data['ids'][0]
    metadata = data['metadatas'][0]
    content = data['documents'][0]
    embedding = data['embeddings'][0] # Тот самый вектор!

    print(f"\n{'='*40}")
    print(f"🆔 ID документа в базе: {doc_id}")
    print(f"{'='*40}")

    print("\n📂 1. МЕТАДАННЫЕ (То, по чему мы фильтруем):")
    print(json.dumps(metadata, indent=4, ensure_ascii=False))

    print("\n📄 2. ТЕКСТ (То, что читает LLM):")
    print("-" * 20)
    print(content)
    print("-" * 20)

    print("\n🧮 3. ВЕКТОР (Как это видит компьютер):")
    print(f"Всего измерений (чисел): {len(embedding)}")
    print(f"Первые 10 чисел: {embedding[:10]}")
    print("... и еще 1000+ таких же чисел.")

    print(f"\n{'='*40}")
    print("✅ Итог: Данные лежат корректно.")

if __name__ == "__main__":
    main()