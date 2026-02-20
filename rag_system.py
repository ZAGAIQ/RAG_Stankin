'''RAG система.
Поисковой движок для нашего бота, работает на gemini-2.5-flash-lite'''

import os
import sys
from operator import itemgetter
from dotenv import load_dotenv
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from chroma_db_operations.db_manager import get_db_connection

load_dotenv()

# --- НАСТРОЙКИ OPENROUTER ---
# Вставь сюда свой ключ или используй переменную окружения
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 

# --- КЛАСС-САНИТАЙЗЕР ---
class SafeRetriever(BaseRetriever):
    """
    Обертка, которая удаляет пустые документы ИЗ ВЫДАЧИ базы,
    прежде чем они попадут в Реранкер и сломают его.
    """
    base_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. Дергаем документы из базы (как раньше)
        docs = self.base_retriever.invoke(query)
        
        valid_docs = []
        for doc in docs:
            # 2. ПРОВЕРЯЕМ: Если контент - не строка или пустой, выкидываем
            if doc.page_content and isinstance(doc.page_content, str) and doc.page_content.strip():
                valid_docs.append(doc)
            else:
                print(f"Warning: Удален битый документ! Meta: {doc.metadata}")
                continue
                
        return valid_docs
    

def get_rag_chain():

    vectorstore = get_db_connection()

    # --- ОПИСАНИЕ МЕТАДАННЫХ ---
    metadata_field_info = [
        AttributeInfo(
            name="source_type",
            description="Категория данных. Выбирай 'Таблица' для поиска фактов, цифр, экзаменов, предметов, баллов, цен и статистики. Выбирай 'Подкаст' только для поиска мнений, советов и обсуждений. Если вопрос содержит 'балл', 'предметы', 'экзамены', 'цена', 'стоимость', 'места', 'количество', 'код' -> Выбирай 'Таблица'",
            type="string",
        ),
        AttributeInfo(
            name="program_code",
            description="Код направления или профиля. Может быть в формате XX.XX.XX (например, 09.03.01) или более детальном XX.XX.XX.XX (например, 09.03.01.03). Используй этот фильтр, если в запросе есть точный цифровой код. Используй точное совпадение (eq).",
            type="string",
        ),
        AttributeInfo(
            name="subjects",
            description="Список экзаменов (ЕГЭ - Единый Государственный Экзамен), необходимых для поступления. Строка с перечислением предметов через запятую с большой буквы (например: 'Информатика, Математика, Русский' или 'Физика, Математика, Русский'). Используй для фильтрации по конкретным предметам, которые сдал абитуриент.",
            type="string",
        ),
        AttributeInfo(
            name="price_rf",
            description="Стоимость обучения за один учебный семестр для граждан РФ в рублях. Это для тех, кто поступает платно. Целое число (например, 182100). Используй этот фильтр для поиска программ по бюджету или порсто консультации (например: 'дешевле 200000' или 'дороже 300000').",
            type="integer",
        ),
        AttributeInfo(
            name="price_in",
            description="Стоимость обучения за один учебный семестр для иностранных граждан в рублях. Целое число (например, 190000). Используй этот фильтр, если пользователь ищет цену для иностранцев, граждан СНГ или 'не для РФ'.",
            type="integer",
        ),
        AttributeInfo(
            name="p_rf_places",
            description="Количество платных мест для граждан РФ. Целое число. Используй этот фильтр, если пользователь спрашивает про количество мест на платной основе для россиян.",
            type="integer",
        ),
        AttributeInfo(
            name="p_in_places",
            description="Количество платных мест для иностранных граждан. Целое число. Используй этот фильтр, если вопрос про количество мест для иностранцев.",
            type="integer",
        ),
        AttributeInfo(
            name="b_places",
            description="Количество бюджетных (бесплатных) мест. Целое число. Используй этот фильтр для поиска программ, где есть бюджетные места (например: 'сколько мест на бюджет?').",
            type="integer",
        ),
        AttributeInfo(
            name="score_last",
            description="Проходной балл, балл ЕГЭ. Целое число (например, 245). Используй для фильтрации по сложности поступления (например: 'баллы ниже 200' или 'реально ли поступить с 230', 'самый низкий проходной балл').",
            type="integer",
        ),
        AttributeInfo(
            name="form",
            description="Форма обучения. Строго одно из значений: 'очная', 'заочная'. Используй, если пользователь уточняет график или режим учебы.",
            type="string",
        ),
        AttributeInfo(
            name="level",
            description="Уровень образования. Строго одно из значений: 'Бакалавриат', 'Специалитет', 'Магистратура', 'Аспирантура'. Используй для выбора конкретной ступени высшего образования.",
            type="string",
        ),
    ]

    document_content_description = "Информация об образовательных программах и правилах приема в университет МГТУ СТАНКИН, включая данные о проходных баллах, стоимости обучения, количестве бюджетных мест и вступительных экзаменах."

    # --- ПОДКЛЮЧЕНИЕ LLM ЧЕРЕЗ OPENROUTER ---
    # Мы используем класс ChatOpenAI, но меняем base_url.
    
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0, # 0 означает строгую логику
    )

    # --- СОЗДАНИЕ SELF-QUERY RETRIEVER ---
    base_retriever = SelfQueryRetriever.from_llm(
        llm,                                
        vectorstore,                        
        document_content_description,       
        metadata_field_info,                
        verbose=True,
        enable_limit=True,
        search_kwargs={"k": 30}
    )

    safe_retriever = SafeRetriever(base_retriever=base_retriever)

    # RERANKER
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=5) # Оставляем топ-5
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=safe_retriever
    )

    # --- ВНЕДРЕНИЕ ПОРОГА  ---
    retriever_runnable = itemgetter("input") | compression_retriever

    system_prompt = (
        "Ты — ассистент приемной комиссии МГТУ СТАНКИН. "
        "Твоя задача — помочь абитуриенту, используя информацию из контекста ниже.\n"
        "1. Внимательно прочитай контекст. Если там есть ответ на вопрос (прямой или косвенный) — ответь.\n"
        "2. Если контекст содержит информацию о направлении (например, робототехника), расскажи о нём, даже если там нет точного ответа про 'разработку роботов'.\n"
        "3. Если информации совсем нет — честно признайся.\n"
        "4. Не выдумывай цифры (цены, баллы).\n\n"
        "ПРАВИЛА ОФОРМЛЕНИЯ:\n"
        "1. НЕ ИСПОЛЬЗУЙ символы '*' для списков. Вместо них используй обычное тире '-' или точку.\n"
        "2. НЕ ИСПОЛЬЗУЙ жирный шрифт через '**'. Используй только HTML-теги <b>текст</b> для выделения важного.\n"
        "3. Коды направлений выделяй тегом <code>код</code>.\n"
        "4. Ответ должен быть чистым, без Markdown-разметки.\n\n"
        "Контекст:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # LLM для генерации ответа
    llm_generation = ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3, # Чуть-чуть креативности для вежливости
    )

    # Цепочка: Документы -> Промпт -> LLM
    combine_docs_chain = create_stuff_documents_chain(llm_generation, prompt)
    
    # Собираем всё вместе. create_retrieval_chain сама прокинет input в retriever_runnable
    rag_chain = create_retrieval_chain(retriever_runnable, combine_docs_chain)

    return rag_chain


# def main():
#     # Инициализируем цепочку
#     rag_chain = get_rag_chain()
    
#     while True:
#         query = input("\nВопрос: ")
#         # Проверка на выход
#         if query.lower() in ['q', 'exit', 'quit']: 
#             break
        
#         try:
#             print("Поиск и фильтрация...")
            
#             # Запускаем RAG
#             response = rag_chain.invoke({"input": query})
            
#             # Выводим ответ LLM
#             print(f"\nБот:\n{response['answer']}")
            
#             # --- БЛОК ОТЛАДКИ (ИСПРАВЛЕННЫЙ) ---
#             print("\n" + "-"*10 + " Документы, прошедшие порог " + "-"*10)
            
#             context_docs = response.get('context', [])
            
#             if not context_docs:
#                 print("Все документы были отсеяны фильтром (или ничего не найдено).")
#             else:
#                 for i, doc in enumerate(context_docs):
#                     # 1. Получаем сырой скор
#                     raw_score = doc.metadata.get('relevance_score')
                    
#                     # 2. Безопасное форматирование
#                     if isinstance(raw_score, (int, float)):
#                         score_display = f"{raw_score:.4f}"
#                     else:
#                         score_display = "N/A" # Если скора нет или он None
                    
#                     # 3. Получаем остальные поля безопасно
#                     source = doc.metadata.get('source_type', 'Неизвестно')
#                     code = doc.metadata.get('program_code', 'Без кода')
                    
#                     # 4. Вывод без ошибки
#                     print(f"#{i+1} [Score: {score_display}] {source} | {code}")

#         except Exception as e:
#             print(f"Ошибка при обработке запроса: {e}")

# if __name__ == "__main__":
#     main()