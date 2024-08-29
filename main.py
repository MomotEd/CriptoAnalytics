from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import asyncio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import timedelta

from starlette.responses import JSONResponse

app = FastAPI()

# Указываем путь к шаблонам
templates = Jinja2Templates(directory="templates")

async def fetch_data(exchange, symbol, timeframe='1s', limit=168*60*60):  # 168 часов = 7 дней
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

    # Получаем список всех тикеров, оканчивающихся на /USDT
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
            merged_data = pd.merge_asof(altcoin_data, btc_data[['timestamp', 'direction', 'close']],
                                        on='timestamp', suffixes=('', '_btc'))

            # Отбираем строки, где направление альткоина совпадает с направлением биткоина
            matching_data = merged_data[merged_data['direction'] == merged_data['direction_btc']]

            if not matching_data.empty:
                # Находим задержку и добавляем время изменения
                def calculate_time_delay_and_times(row, btc_data):
                    btc_change_time = btc_data.loc[btc_data['timestamp'] <= row['timestamp'], 'timestamp'].max()
                    delay = (row['timestamp'] - btc_change_time).total_seconds()
                    return pd.Series({'time_delay': delay, 'btc_time': btc_change_time, 'altcoin_time': row['timestamp']})

                matching_data = matching_data.join(
                    matching_data.apply(calculate_time_delay_and_times, axis=1, btc_data=btc_data)
                )

                # Группируем данные по направлениям движения
                result = matching_data.groupby(['direction_btc']).agg({
                    'time_delay': 'mean',
                    'btc_time': 'first',
                    'altcoin_time': 'first',
                    'direction': lambda x: x.mode()[0]
                }).reset_index()

                # Добавляем результат в список
                matching_results.append({
                    'altcoin_pair': symbol,
                    'result': result.to_dict(orient='records'),
                    'full_data': matching_data[['timestamp', 'close', 'direction', 'btc_time', 'altcoin_time']].to_dict(orient='records')
                })

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Закрываем соединение с биржей
    await exchange.close()

    # Рендерим HTML-шаблон с переданными данными
    return templates.TemplateResponse("index.html", {"request": request, "data": matching_results})

@app.get("/graph", response_class=HTMLResponse)
async def graph(request: Request):
    exchange = ccxt.binance()

    # Получаем данные по BTC/USDT
    btc_data = await fetch_data(exchange, 'BTC/USDT')

    if btc_data is None:
        return JSONResponse(content={"error": "BTC/USDT data could not be fetched"})

    # Обработка данных биткоина
    btc_data = get_price_changes(btc_data)

    # Устанавливаем диапазон дат для последней недели
    end_date = btc_data['timestamp'].max()
    start_date = end_date - timedelta(days=7)

    # Получаем список всех тикеров, оканчивающихся на /USDT
    markets = await exchange.load_markets()
    ticker_symbols = [symbol for symbol in markets.keys() if symbol != 'BTC/USDT' and symbol.endswith('/USDT')]

    # Асинхронно получаем данные по всем тикерам
    tasks = [fetch_data(exchange, symbol) for symbol in ticker_symbols]
    results = await asyncio.gather(*tasks)

    graphs = []
    correlation_threshold = 0.8  # Порог корреляции для фильтрации

    for symbol, altcoin_data in zip(ticker_symbols, results):
        if altcoin_data is None:
            continue

        try:
            altcoin_data = get_price_changes(altcoin_data)

            # Приводим временные ряды биткоина и альткоина к одному временному интервалу
            merged_data = pd.merge_asof(altcoin_data[['timestamp', 'close']], btc_data[['timestamp', 'close']], on='timestamp', suffixes=('', '_btc'))

            # Вычисляем коэффициент корреляции
            correlation = merged_data['close'].corr(merged_data['close_btc'])

            if correlation >= correlation_threshold:
                # Создаем графики для биткоина и альткоина, если они статистически похожи
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # График цены биткоина
                fig.add_trace(
                    go.Scatter(x=btc_data['timestamp'], y=btc_data['close'], mode='lines', name='BTC/USDT'),
                    secondary_y=False,
                )

                # График цены альткоина
                fig.add_trace(
                    go.Scatter(x=altcoin_data['timestamp'], y=altcoin_data['close'], mode='lines', name=symbol),
                    secondary_y=True,
                )

                # Настройка осей с фиксированным диапазоном для последней недели
                fig.update_xaxes(title_text="Время", range=[start_date, end_date])
                fig.update_yaxes(title_text="Цена BTC/USDT", secondary_y=False)
                fig.update_yaxes(title_text=f"Цена {symbol}", secondary_y=True)

                # Настройка заголовка
                fig.update_layout(
                    title_text=f"Сравнение цен BTC/USDT и {symbol} (Корреляция: {correlation:.2f})",
                    height=600,
                )

                graph_html = fig.to_html(full_html=False)
                graphs.append({"symbol": symbol, "graph": graph_html})

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Закрываем соединение с биржей
    await exchange.close()

    return templates.TemplateResponse("graph.html", {"request": request, "graphs": graphs})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
