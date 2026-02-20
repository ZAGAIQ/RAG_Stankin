'''Менеджер векторной базы данных. 
С помощью данного скрипта мы подключаемся к бд и удаляем её если это необходимо.'''

import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

def get_db_connection() -> Chroma:
    """
    Подключается к базе данных. Если базы нет - она будет создана автоматически 
    при первом добавлении документов.
    """
    print(f"[DB Manager] Инициализация эмбеддингов ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"[DB Manager] Подключение к ChromaDB ({CHROMA_PATH})...")
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    return db

def reset_database():
    """Полностью удаляет базу данных."""
    print(f"[DB Manager] Запрос на удаление базы: {CHROMA_PATH}")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("[DB Manager] База данных успешно и полностью удалена.")
    else:
        print("[DB Manager] База не найдена.")


# if __name__ == "__main__":
#     reset_database()