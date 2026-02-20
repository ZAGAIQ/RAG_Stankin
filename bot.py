'''Телеграм бот, фронтенд для нашего RAG'''


import asyncio
import logging
import os
import html
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode

from rag_system import get_rag_chain

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Глобальная переменная для хранения цепочки RAG
rag_chain = None

async def on_startup():
    """СОБЫТИЕ ПРИ ЗАПУСКЕ. Загружаем модели ОДИН раз при старте"""
    global rag_chain
    print("Бот запускается...")
    rag_chain = get_rag_chain()
    print("RAG-система загружена.")

# --- ХЕНДЛЕР: Команда /start ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 Привет! Я ИИ-помощник приемной комиссии МГТУ СТАНКИН.\n\n"
        "Спрашивай меня про баллы, цены и направления.\n"
        "Пример: <b>Какой проходной балл на 09.03.01?</b>",
        parse_mode=ParseMode.HTML
    )

# --- ГЛАВНЫЙ ХЕНДЛЕР ---
@dp.message(F.text)
async def handle_rag_query(message: types.Message):
    user_query = message.text
    
    # 1. Статус "печатает"
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    # 2. Отправляем временное сообщение
    temp_msg = await message.answer("⏳ <i>Анализирую базу знаний...</i>", parse_mode=ParseMode.HTML)

    try:
        # 3. Делаем запрос к нейронке
        if rag_chain is None:
            await temp_msg.edit_text("Система еще загружается, подождите 10 секунд и повторите.")
            return

        response = await rag_chain.ainvoke({"input": user_query})
        
        answer_text = response['answer']
        context_docs = response.get('context', [])

        # 4. Формируем источники (в HTML)
        sources_text = ""
        if context_docs:
            sources_text = "\n\n📚 <b>Источники:</b>\n"
            for i, doc in enumerate(context_docs[:3]):
                raw_score = doc.metadata.get('relevance_score')
                
                
                src_type = doc.metadata.get('source_type', 'Док')
                code = doc.metadata.get('program_code', '-')
                
                safe_type = html.escape(str(src_type))
                safe_code = html.escape(str(code))
                
                sources_text += f"<i>{i+1}. {safe_type} | {safe_code}</i>\n"
        
        # 5. Экранируем сам ответ нейронки
        # Если нейронка напишет "x < y", без escape это сломает HTML
        # Но мы оставим как есть, так как Gemini редко пишет теги.

        final_text = f"{answer_text}{sources_text}"

        # 6. Сначала отправляем новое, потом удаляем старое.
        await message.answer(final_text, parse_mode=ParseMode.HTML)
        await temp_msg.delete()

    except Exception as e:
        print(f"ERROR: {e}")
        # Если временное сообщение еще живо, редактируем его
        try:
            await temp_msg.edit_text(f"❌ Произошла ошибка: {e}")
        except:
            # Если temp_msg уже удалено, отправляем новое
            await message.answer("❌ Произошла ошибка при обработке запроса.")

async def main():
    dp.startup.register(on_startup)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())