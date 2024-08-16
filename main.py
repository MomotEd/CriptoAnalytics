from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import asyncio

app = FastAPI()

# Указываем путь к шаблонам
templates = Jinja2Templates(directory="templates")


async def fetch_data(exchange, symbol, timeframe='1h', limit=100):
    """Асинхронное получение данных OHLCV для символа"""
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_price_changes(data):
    """Определение изменений в цене и направления движения (рост/спад)"""
    data['price_change'] = data['close'].pct_change()
    data['direction'] = np.where(data['price_change'] > 0, 'рост', 'спад')
    return data

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    exchange = ccxt.binance()

    # Получаем данные по BTC/USDT
    btc_data = await fetch_data(exchange, 'BTC/USDT')

    if btc_data is None:
        return templates.TemplateResponse("index.html", {"request": request, "data": []})

    # Обработка данных биткоина
    btc_data = get_price_changes(btc_data)

    # Получаем список всех тикеров
    markets = await exchange.load_markets()
    ticker_symbols = [symbol for symbol in markets.keys() if symbol != 'BTC/USDT' and symbol.endswith('/USDT')]

    # Асинхронно получаем данные по всем тикерам
    tasks = [fetch_data(exchange, symbol) for symbol in ticker_symbols]
    results = await asyncio.gather(*tasks)

    matching_results = []

    for symbol, altcoin_data in zip(ticker_symbols, results):
        if altcoin_data is None:
            continue

        try:
            altcoin_data = get_price_changes(altcoin_data)

            # Сводим данные по времени и направлению движения
            merged_data = pd.merge_asof(altcoin_data, btc_data[['timestamp', 'direction']],
                                        on='timestamp', suffixes=('', '_btc'))

            # Отбираем только те строки, где направление альткоина совпадает с направлением биткоина
            matching_data = merged_data[merged_data['direction'] == merged_data['direction_btc']]

            if not matching_data.empty:
                # Находим среднюю задержку для совпадающих направлений
                def calculate_time_delay(row, btc_data):
                    btc_change_time = btc_data.loc[btc_data['timestamp'] <= row['timestamp'], 'timestamp'].max()
                    return (row['timestamp'] - btc_change_time).total_seconds()

                matching_data.loc[:, 'time_delay'] = matching_data.apply(calculate_time_delay, axis=1, btc_data=btc_data)

                # Группируем данные по направлениям движения
                result = matching_data.groupby(['direction_btc']).agg({
                    'time_delay': 'mean',
                    'direction': lambda x: x.mode()[0]
                }).reset_index()

                # Добавляем результат в список
                matching_results.append({
                    'altcoin_pair': symbol,
                    'result': result.to_dict(orient='records')
                })

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Закрываем соединение с биржей
    await exchange.close()

    # Рендерим HTML-шаблон с переданными данными
    return templates.TemplateResponse("index.html", {"request": request, "data": matching_results})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
