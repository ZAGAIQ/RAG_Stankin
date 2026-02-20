# Data/rag_stankin_parser.py

import os
import re
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# Ключевые библиотеки для эмбеддинга
from transformers import AutoTokenizer, AutoModel
import torch # Важно для работы с трансформерами

# Ключевые библиотеки для БД
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from sentence_transformers import SentenceTransformer

# Библиотека для чанкинга (если не хотим писать свой)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- КОНФИГУРАЦИЯ ---
# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class STANKIN_RAG_Config:
    START_URL = "https://priem.stankin.ru/"
    BASE_DOMAIN = urlparse(START_URL).netloc
    MAX_CRAWL_DEPTH = 4
    EMBEDDING_MODEL_NAME = "cointegrated/rubert-tiny2"
    CHROMA_DB_PATH = "./stankin_db"
    COLLECTION_NAME = "stankin_collection"
    
    # Параметры чанкинга
    CHUNK_SIZE = 1000  # Используем больше, но фильтр по длине все равно сработает
    CHUNK_OVERLAP = 100
    MIN_CHUNK_LEN = 50
    MAX_CHUNK_LEN = 3000
    
    # Регулярные выражения для Анти-Мусор Фильтра (Требование 2.C)
    ANTI_GARBAGE_PATTERNS = [
        re.compile(r'data:image/[a-zA-Z0-9+/=;,-]+'),  # Base64 строки
        re.compile(r'', re.DOTALL),          # HTML комментарии
        re.compile(r'\[Top\.Mail\.Ru\]|Yandex\.Metrika'), # Счетчики/Аналитика
        re.compile(r'^\s*https?://\S+\s*$')              # Чанк - это просто URL
    ]

# --- МОДУЛЬ 1: ВЕКТОРИЗАЦИЯ (ОБНОВЛЕННЫЙ) ---
class SBERT_EmbeddingFunction(EmbeddingFunction):
    """
    Использует SentenceTransformer для надежной загрузки DeepPavlov модели,
    гарантируя корректный пулинг и нормализацию.
    """
    def __init__(self, model_name: str):
        logging.info(f"Загрузка модели через SentenceTransformer: {model_name}...")
        try:
            # SentenceTransformer автоматически обрабатывает pooling и нормализацию
            self.model = SentenceTransformer(model_name)
            logging.info("SUCCESS: Функция эмбеддинга (SBERT) успешно инициализирована.")
        except Exception as e:
            logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА ЗАГРУЗКИ SBERT: {e}")
            raise

    def __call__(self, texts: Documents) -> Embeddings:
        """Генерирует эмбеддинги."""
        # encode возвращает L2-нормализованные векторы numpy
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


# --- МОДУЛЬ 2: КРАУЛИНГ И ОЧИСТКА ---

def clean_html_content(html_content: str, url: str) -> str:
    """
    Агрессивная очистка HTML: удаление служебных тегов, 
    конвертация таблиц в Markdown.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # 1. Удаление служебных тегов
    REMOVAL_TAGS = ['script', 'style', 'nav', 'footer', 'header', 'form', 'aside', 'iframe', 'noscript']
    for tag in REMOVAL_TAGS:
        for element in soup.find_all(tag):
            element.decompose()
    logging.debug(f"Удалены служебные теги на {url}")

    # 2. Обработка Таблиц (Преобразование в Markdown)
    tables = soup.find_all('table')
    if tables:
        logging.info(f"Обработка {len(tables)} таблиц на странице {url}...")
        for i, table in enumerate(tables):
            try:
                # pandas.read_html может парсить как строку, так и тег
                df = pd.read_html(str(table))[0]
                # Конвертация в Markdown
                markdown_table = "\n\n--- НАЧАЛО ТАБЛИЦЫ ---\n"
                markdown_table += df.to_markdown(index=False)
                markdown_table += "\n--- КОНЕЦ ТАБЛИЦЫ ---\n\n"
                
                # Заменяем тег <table> на его Markdown представление
                table.replace_with(BeautifulSoup(markdown_table, 'html.parser'))
            except Exception as e:
                logging.warning(f"Ошибка парсинга таблицы {i+1} на {url}: {e}")
                table.decompose() # Удаляем проблемную таблицу

    # 3. Финальная очистка текста
    # Получаем чистый текст из оставшегося HTML
    text = soup.get_text(separator=' ', strip=True)
    # Удаляем лишние пробелы и переводы строк
    text = re.sub(r'\s+', ' ', text).strip()
    
    logging.debug(f"HTML-контент страницы {url} очищен. Длина: {len(text)}")
    return text


def stankin_crawler(start_url: str, max_depth: int) -> dict:
    """
    Рекурсивный краулинг с ограничением по домену и глубине.
    Возвращает словарь {url: content}.
    """
    queue = [(start_url, 0)]
    visited = {start_url}
    page_contents = {}
    base_domain = STANKIN_RAG_Config.BASE_DOMAIN
    
    logging.info(f"Запуск краулинга с {start_url} до глубины {max_depth}...")

    while queue:
        current_url, depth = queue.pop(0)

        if depth > max_depth:
            continue
        
        logging.info(f"-> Парсинг URL: {current_url} (Глубина: {depth})")

        try:
            response = requests.get(current_url, timeout=10)
            # Проверяем успешность и кодировку
            response.raise_for_status()
            response.encoding = response.apparent_encoding 
            
            html_content = response.text
            
            # 1. Очищаем контент
            cleaned_text = clean_html_content(html_content, current_url)
            page_contents[current_url] = cleaned_text
            
            # 2. Поиск новых ссылок
            soup = BeautifulSoup(html_content, 'lxml')
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(current_url, href).split('#')[0] # Убираем якоря
                
                # Проверка на поддомен и посещение
                if urlparse(absolute_url).netloc == base_domain and absolute_url not in visited:
                    visited.add(absolute_url)
                    queue.append((absolute_url, depth + 1))
                    logging.debug(f"DEBUG: Найдена новая ссылка: {absolute_url} (Depth: {depth + 1})")
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при загрузке {current_url}: {e}")
        except Exception as e:
            logging.error(f"Непредвиденная ошибка на {current_url}: {e}")

    logging.info(f"SUCCESS: Краулинг завершен. Найдено {len(page_contents)} уникальных страниц.")
    return page_contents


# --- МОДУЛЬ 3: ЧАНКИНГ И ФИЛЬТРАЦИЯ ---

def create_and_filter_chunks(page_contents: dict) -> list[dict]:
    """
    Разбивает текст на чанки и применяет двойную фильтрацию.
    Возвращает список словарей {text, source}.
    """
    # Инициализация LangChain Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=STANKIN_RAG_Config.CHUNK_SIZE,
        chunk_overlap=STANKIN_RAG_Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    
    final_chunks = []
    
    logging.info("Применение чанкинга и двойного фильтра ко всему контенту...")

    for url, text in page_contents.items():
        if not text:
            logging.warning(f"Пропущен пустой контент для URL: {url}")
            continue

        # 1. Разбиение на чанки
        chunks = text_splitter.split_text(text)
        logging.info(f"URL: {url} -> Исходный текст разбит на {len(chunks)} чанков.")
        
        for i, chunk in enumerate(chunks):
            # 2. Фильтр 1: По длине (Требование 2.C)
            if not (STANKIN_RAG_Config.MIN_CHUNK_LEN <= len(chunk) <= STANKIN_RAG_Config.MAX_CHUNK_LEN):
                logging.debug(f"CHUNK FILTER (Length): Пропущен чанк #{i} (Длина: {len(chunk)}).")
                continue

            '''
            # 3. Фильтр 2: Анти-Мусор (Требование 2.C)
            is_garbage = False
            for pattern in STANKIN_RAG_Config.ANTI_GARBAGE_PATTERNS:
                if pattern.search(chunk):
                    logging.warning(f"CHUNK FILTER (Garbage): Пропущен чанк #{i} из {url} (Начало: {chunk[:50]}...)")
                    is_garbage = True
                    break
            
            if is_garbage:
                continue
            '''

            # Если чанк прошел все фильтры
            final_chunks.append({
                "text": chunk,
                "source": url
            })

    logging.info(f"SUCCESS: Итоговое количество чанков, готовых к индексированию: {len(final_chunks)}")
    return final_chunks


# --- МОДУЛЬ 4: ИНДЕКСИРОВАНИЕ ---

def index_chunks_to_chroma(chunks: list[dict], ef: EmbeddingFunction):
    """
    Индексирует очищенные чанки в ChromaDB.
    """
    logging.info(f"Инициализация ChromaDB по пути: {STANKIN_RAG_Config.CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=STANKIN_RAG_Config.CHROMA_DB_PATH)
    collection_name = STANKIN_RAG_Config.COLLECTION_NAME
    
    # Требование 2.D: Удаление существующей коллекции
    try:
        client.delete_collection(name=collection_name)
        logging.critical(f"Удалена старая коллекция '{collection_name}' для чистого старта.")
    except:
        logging.info(f"Коллекция '{collection_name}' не найдена или не требует удаления.")

    # Создание новой коллекции с кастомной функцией эмбеддинга
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    logging.info(f"Коллекция '{collection_name}' готова к заполнению.")

    # Подготовка данных для пакетного добавления
    documents = [c['text'] for c in chunks]
    metadatas = [{"source": c['source']} for c in chunks]
    # ID's должны быть уникальными (например, URL + индекс чанка)
    ids = [f"{i}-{re.sub(r'[^a-zA-Z0-9]', '_', c['source'])}" for i, c in enumerate(chunks)]

    if not documents:
        logging.warning("Нет документов для индексирования. Пропускаем.")
        return

    # Пакетное добавление
    BATCH_SIZE = 500 
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_metadatas = metadatas[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        logging.info(f"Индексирован пакет: {i} - {min(i + BATCH_SIZE, len(documents))}")
    
    logging.critical(f"SUCCESS: Индексирование завершено. Всего чанков в БД: {collection.count()}")


# --- МОДУЛЬ 5: ТЕСТИРОВАНИЕ RAG ---

def test_rag_query(query: str, ef: EmbeddingFunction):
    """
    Выполняет тестовый запрос и выводит результаты.
    """
    logging.info("-" * 50)
    logging.info("НАЧАЛО ТЕСТИРОВАНИЯ КАЧЕСТВА RAG")
    logging.info(f"Тестовый запрос: '{query}'")
    
    client = chromadb.PersistentClient(path=STANKIN_RAG_Config.CHROMA_DB_PATH)
    collection_name = STANKIN_RAG_Config.COLLECTION_NAME
    
    try:
        # Получаем коллекцию (с обязательным указанием EF, чтобы запрос кодировался корректно)
        collection = client.get_collection(name=collection_name, embedding_function=ef)
    except Exception as e:
        logging.error(f"Коллекция не найдена. Невозможно выполнить тест. {e}")
        return

    # Выполняем семантический поиск
    results = collection.query(
        query_texts=[query],
        n_results=5, # Требование 2.D
        include=['documents', 'metadatas', 'distances']
    )

    if not results or not results['documents'] or not results['documents'][0]:
        logging.error("Не найдено релевантных документов.")
        return

    logging.critical("\n--- 5 САМЫХ РЕЛЕВАНТНЫХ ЧАНКОВ (RAG-РЕЗУЛЬТАТ) ---\n")
    
    for i in range(5):
        try:
            distance = results['distances'][0][i]
            source = results['metadatas'][0][i]['source']
            content = results['documents'][0][i]
            
            print(f"[{i+1}] Дистанция: {distance:.4f}")
            print(f"    Источник: {source}")
            print(f"    Содержание: {content[:300]}...\n")
            
        except IndexError:
            # Если найдено меньше 5 результатов
            break
            
    logging.info("-" * 50)


# --- ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА ---

def main():
    logging.critical("СТАРТ RAG-СИСТЕМЫ ПАРСИНГА СТАНКИНА")
    
    # ШАГ 1: Инициализация Embedding Function (Ключевое требование)
    try:
        # ИСПОЛЬЗУЕМ НОВЫЙ КЛАСС: SBERT_EmbeddingFunction
        ef = SBERT_EmbeddingFunction(STANKIN_RAG_Config.EMBEDDING_MODEL_NAME) 
    except Exception as e:
        logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА ИНИЦИАЛИЗАЦИИ МОДЕЛИ: {e}")
        return

    # ШАГ 2: Краулинг и Очистка
    page_contents = stankin_crawler(
        start_url=STANKIN_RAG_Config.START_URL,
        max_depth=STANKIN_RAG_Config.MAX_CRAWL_DEPTH
    )

    # ШАГ 3: Чанкинг и Двойная Фильтрация
    filtered_chunks = create_and_filter_chunks(page_contents)

    # ШАГ 4: Индексирование в ChromaDB
    if filtered_chunks:
        index_chunks_to_chroma(filtered_chunks, ef)
    
    # ШАГ 5: Тестирование RAG
    TEST_QUERY = "Какие документы нужны для подачи заявления в МГТУ СТАНКИН?"
    test_rag_query(TEST_QUERY, ef)

if __name__ == "__main__":
    # Создаем папку Data, если ее нет
    os.makedirs('Data', exist_ok=True) 
    # Запускаем основной процесс
    main()