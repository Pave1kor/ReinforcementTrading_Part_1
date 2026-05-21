import asyncio
import aiohttp
import pandas as pd
import aiomoex
from datetime import datetime

async def fetch_moex_candles(ticker, start_date, end_date, interval=24):
    """
    Асинхронное получение свечей с MOEX ISS.
    
    Параметры:
        ticker (str): Тикер инструмента, например, 'SBER', 'GAZP', 'IMOEX'.
        start_date (str, datetime): Начальная дата в формате 'YYYY-MM-DD'.
        end_date (str, datetime): Конечная дата в формате 'YYYY-MM-DD'.
        interval (int): Минутный интервал свечи (1, 10, 60, 24(день)).
    
    Возвращает:
        pandas.DataFrame: Датафрейм с колонками 'open', 'close', 'high', 'low', 'volume'.
    """
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')

    async with aiohttp.ClientSession() as session:
        # Данные будут загружены сразу в pandas DataFrame
        data = await aiomoex.get_market_candles(
            session=session,
            security=ticker,
            interval=interval,
            start=start_date,
            end=end_date
        )
    
    df = pd.DataFrame(data)
    
    # Важно: сортируем по дате и устанавливаем её как индекс
    if not df.empty:
        df['begin'] = pd.to_datetime(df['begin'])
        df = df.sort_values('begin')
        df.set_index('begin', inplace=True)
        # Выбираем нужные колонки
        df = df[['open', 'close', 'high', 'low', 'volume']]
        # Убираем строки с пропущенными ценами
        df = df.dropna(subset=['open', 'close', 'high', 'low'])
        print(f"✅ Успешно загружено {len(df)} свечей для {ticker}.")
    else:
        print(f"⚠️ Не найдено данных для {ticker} в указанном диапазоне.")
    
    return df

# Пример использования:
async def main():
    # Скачаем дневные свечи Сбербанка за 2023 год
    df_sber = await fetch_moex_candles(
        ticker='SBER',
        start_date='2024-01-01',
        end_date='2025-12-31',
        interval=10  # Дневные свечи
    )
    
    # Сохраняем в CSV
    if not df_sber.empty:
        df_sber.to_csv('data/SBER_test_2023_daily.csv')
        print("Данные сохранены в 'SBER_2023_daily.csv'")
        print(df_sber.head())

# Запуск асинхронной функции
if __name__ == "__main__":
    asyncio.run(main())