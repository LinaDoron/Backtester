# main.py
import asyncio
import logging
from aiogram import Bot, Dispatcher
from config import BOT_TOKEN
from bot.handlers import router
from muscle.engine import load_data_to_cache

# Включаем логирование, чтобы видеть ошибки в консоли
logging.basicConfig(level=logging.INFO)

async def main():
    print("Запуск...")
    
    # ЗАГРУЗКА ДАННЫХ
    load_data_to_cache()

    # Инициализация бота и диспетчера
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    # Подключаем наши обработчики
    dp.include_router(router)

    # Сбрасываем старые апдейты (если бот был оффлайн) и запускаем
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
