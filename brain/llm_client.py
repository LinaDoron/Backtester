# brain/llm_client.py
import json
import re
from openai import AsyncOpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

MODEL = LLM_MODEL

FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n?[ \t]*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = FENCE_RE.match(text)
    return match.group(1) if match else text


def _clean_code(code: str) -> str:
    """Модель иногда кладет в поле code маркдаун-блок ```python ... ``` или голую строку 'python'."""
    code = _strip_fence(code.strip())
    lines = code.split('\n')
    if lines and lines[0].strip().lower() in ('python', 'py'):
        lines = lines[1:]
    return '\n'.join(lines).strip() + '\n'


def _extract_json(raw: str) -> str:
    text = _strip_fence(raw.strip())
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end < start:
        return text
    return text[start:end + 1]

async def generate_strategy_code(user_text: str) -> dict:
    """
    Отправляет запрос Дипсику, получает ответ и парсит JSON.
    Возвращает словарь: {'config': {...}, 'code': '...'}
    """
    from brain.prompts import SYSTEM_PROMPT

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            temperature=0.1, # Низкая температура для строгого кода
        )

        raw_response = response.choices[0].message.content.strip()

        # Снимаем только внешнюю обертку ```json ... ```: внутри поля code бывают свои ```.
        parsed_data = json.loads(_extract_json(raw_response))
        if isinstance(parsed_data, dict) and isinstance(parsed_data.get('code'), str):
            parsed_data['code'] = _clean_code(parsed_data['code'])
        return parsed_data

    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return None
