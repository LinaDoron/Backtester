import asyncio
from aiogram import Router, types
from aiogram.filters import Command
from aiogram.types import FSInputFile
from config import ADMIN_ID
from brain.llm_client import generate_strategy_code
from muscle.engine import run_backtest

# Создаем роутер для обработчиков
router = Router()

# Мидлварь (фильтр) для проверки на админа прямо в функции
def is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID

@router.message(Command("start"))
async def cmd_start(message: types.Message):
    if not is_admin(message.from_user.id):
        return # Игнорим чужих

    await message.answer(
        "На связи. Мозг холодный, мышцы готовы. \n\n"
        "Жду твою гипотезу для бэктеста."
    )

@router.message()
async def handle_hypothesis(message: types.Message):
    if not is_admin(message.from_user.id): return

    status_msg = await message.answer("Обрабатываю гипотезу...")

    # 1. Отправляем текст Дипсику
    llm_response = await generate_strategy_code(message.text)

    if not llm_response or 'code' not in llm_response:
        await status_msg.edit_text("Ошибка: Дипсик не смог сгенерировать код. Попробуй переформулировать.")
        return

    config = llm_response.get('config')
    code_str = llm_response['code']

    await status_msg.edit_text("Код сгенерирован. Запускаю бэктест...")

    # 2. Передаем сгенерированный код в Движок. В потоке, чтобы бот не замирал на время расчета.
    result = await asyncio.to_thread(run_backtest, code_str, config)

    # 3. Обрабатываем ответ Движка
    if result.get("error"):
        await status_msg.edit_text(f"Ошибка экзекуции:\n{result['error']}")
        return

    # 4. Формируем красивый отчет
    report = (
        f"<b>Бэктест завершен!</b>\n\n"
        f"Стартовый депозит: ${config['deposit']:.2f}\n"
        f"Финальный баланс: ${result['final_deposit']:.2f}\n"
        f"Чистая прибыль: <b>${result['total_pnl']:.2f}</b>\n\n"
        f"Макс. просадка (MDD): <b>{result['mdd']:.2f}%</b>\n\n"
        f"Винрейт: <b>{result['winrate']:.1f}%</b>\n"
        f"Сделок всего: {result['total_trades']} ({result['wins']}✅ / {result['losses']}❌)\n"
        f"Закрыто досрочно: {result.get('closed', 0)}\n\n"
        f"Риск на сделку: {config['risk_pct']*100}%\n"
        f"RR: 1 к {config['tp_ratio']}"
    )

    # Отправляем фото графика и удаляем сообщение со статусом
    photo = FSInputFile(result['plot_path'])
    await message.answer_photo(photo=photo, caption=report, parse_mode="HTML")
    await status_msg.delete()
