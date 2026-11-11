import os
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_TG_ID"))

# Любой OpenAI-совместимый API: OpenRouter, DeepSeek и т.д.
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-super-vl:free")

if not BOT_TOKEN or not ADMIN_ID:
    raise ValueError("ОШИБКА: Проверь .env файл, не хватает токена или ADMIN_ID!")
