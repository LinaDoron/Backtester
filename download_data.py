# download_data.py
import yfinance as yf
import os

# Создаем папку, если ее нет
os.makedirs("data/csv", exist_ok=True)

# Активы (Yahoo Finance использует суффикс =X для форекса)
assets = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X"
    # Внимание: Франк - обратная пара. Его Хай = Лой Евро. Мы учтем это в Математике!
}

print("Начинаю загрузку котировок за последние 60 дней (M5)...")

for name, ticker in assets.items():
    print(f"Качаю {name}...")
    # Качаем 5-минутки за 60 дней
    data = yf.download(ticker, interval="5m", period="60d", progress=False)

    # Очищаем данные (убираем мультииндексы Yahoo и лишние колонки)
    data = data.reset_index()

    # Переименовываем колонки для удобства Кабала
    # Yahoo отдает: Datetime, Open, High, Low, Close, Adj Close, Volume
    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
    if 'datetime' in data.columns:
        data = data.rename(columns={'datetime': 'time'})

    # Сохраняем в CSV
    filepath = f"data/csv/{name.lower()}_m5.csv"
    data[['time', 'open', 'high', 'low', 'close']].to_csv(filepath, index=False)
    print(f"Сохранено: {filepath} ({len(data)} свечей)")

print("✅ Все данные успешно скачаны!")
