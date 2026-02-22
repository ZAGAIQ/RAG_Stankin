'''RAG система.
Поисковой движок для нашего бота, работает на gemini-2.5-flash-lite'''

import os
import sys
from operator import itemgetter
from dotenv import load_dotenv
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

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
    
# --- ХРАНИЛИЩЕ ИСТОРИИ ЧАТОВ ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    
    # Ограничиваем историю 4 последними сообщениями (2 пары вопрос-ответ)
    if len(store[session_id].messages) > 4:
        store[session_id].messages = store[session_id].messages[-4:]
        
    return store[session_id]

def get_rag_chain():

    vectorstore = get_db_connection()

    # --- ОПИСАНИЕ МЕТАДАННЫХ ---
    metadata_field_info = [
        AttributeInfo(
            name="source_type",
            description="Категория данных. Выбирай 'Таблица' для поиска фактов, цифр, экзаменов, предметов, баллов, цен и статистики. Выбирай 'Подкаст' только для поиска мнений, советов и обсуждений. Если вопрос содержит 'балл', 'предметы', 'экзамены', 'цена', 'стоимость', 'места', 'количество', 'код' -> Выбирай 'Таблица'. Если пользователь пишет предметы (например: 'сдавал физику', 'куда поступить с химией') -> ОБЯЗАТЕЛЬНО выбирай 'Таблица'.",
            type="string",
        ),
        AttributeInfo(
            name="program_code",
            description="Код направления подготовки в формате XX.XX.XX (например: '09.03.01', '15.03.04'). ВНИМАНИЕ: СТРОГО ЗАПРЕЩАЕТСЯ выдумывать код!",
            type="string",
        ),
        # AttributeInfo(
        #     name="subjects",
        #     description="Список вступительных экзаменов (ЕГЭ). Содержит предметы с большой буквы через запятую. Если пользователь пишет 'сдавал физику', 'с информатикой', 'какие экзамены', ОБЯЗАТЕЛЬНО используй оператор 'contain' (содержит) для поиска по этому полю. Названия предметов пиши с большой буквы (например: 'Физика', 'Химия').",
        #     type="string",
        # ),
        # AttributeInfo(
        #     name="price_rf",
        #     description="Стоимость обучения за один учебный семестр для граждан РФ в рублях. Это для тех, кто поступает платно. Целое число (например, 182100). Используй этот фильтр для поиска программ по бюджету или порсто консультации (например: 'дешевле 200000' или 'дороже 300000').",
        #     type="integer",
        # ),
        # AttributeInfo(
        #     name="price_in",
        #     description="Стоимость обучения за один учебный семестр для иностранных граждан в рублях. Целое число (например, 190000). Используй этот фильтр, если пользователь ищет цену для иностранцев, граждан СНГ или 'не для РФ'.",
        #     type="integer",
        # ),
        # AttributeInfo(
        #     name="p_rf_places",
        #     description="Количество платных мест для граждан РФ. Целое число. Используй этот фильтр, если пользователь спрашивает про количество мест на платной основе для россиян.",
        #     type="integer",
        # ),
        # AttributeInfo(
        #     name="p_in_places",
        #     description="Количество платных мест для иностранных граждан. Целое число. Используй этот фильтр, если вопрос про количество мест для иностранцев.",
        #     type="integer",
        # ),
        # AttributeInfo(
        #     name="b_places",
        #     description="Количество бюджетных (бесплатных) мест. Целое число. Используй этот фильтр для поиска программ, где есть бюджетные места (например: 'сколько мест на бюджет?').",
        #     type="integer",
        # ),
        AttributeInfo(
            name="score_last",
            description="Проходной балл прошлого года. Если абитуриент пишет 'у меня N баллов', 'я набрал N баллов', 'мои баллы N' — ОБЯЗАТЕЛЬНО используй математический оператор 'lte' (меньше или равно), чтобы найти все направления, куда хватает его баллов. Используй фильтры (lt, lte, gt, gte, eq) ТОЛЬКО если в запросе есть КОНКРЕТНАЯ ЦИФРА. Если цифры нет — фильтр не создавай.",
            type="integer",
        ),
        # AttributeInfo(
        #     name="form",
        #     description="Форма обучения. Строго одно из значений: 'очная', 'заочная'. Используй, если пользователь уточняет график или режим учебы.",
        #     type="string",
        # ),
        # AttributeInfo(
        #     name="level",
        #     description="Уровень образования. Строго одно из значений: 'Бакалавриат', 'Специалитет', 'Магистратура', 'Аспирантура'. Используй для выбора конкретной ступени высшего образования.",
        #     type="string",
        # ),
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
        search_kwargs={"k": 50}
    )

    safe_retriever = SafeRetriever(base_retriever=base_retriever)

    # RERANKER
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=25) # Оставляем топ-5
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=safe_retriever
    )

    # --- ИСТОРИЯ ЧАТА: РЕТРИВЕР ---
    
    # Промпт для переформулирования вопроса с учетом истории
    contextualize_q_system_prompt = (
        "Учитывая историю чата и последний вопрос пользователя, "
        "который может ссылаться на контекст из истории чата, "
        "сформулируй самостоятельный вопрос, который можно понять без истории чата. "
        "НЕ отвечай на вопрос, просто переформулируй его при необходимости, "
        "иначе верни его как есть."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Создаем ретривер, который умеет работать с историей
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        compression_retriever,
        contextualize_q_prompt
    )

    system_prompt = (
        "Ты — ассистент приемной комиссии МГТУ СТАНКИН. "
        "Текущая дата и время: {time}. "
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
        MessagesPlaceholder("chat_history"),
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
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    # Оборачиваем в управление историей
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


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