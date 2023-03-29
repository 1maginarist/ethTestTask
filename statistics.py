import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import asyncio
import aiohttp


async def get_square_correlation():
    # Настройка параметров
    endpoint = 'https://api.binance.com/api/v3/klines'
    symbol_eth = 'ETHUSDT'
    symbol_btc = 'BTCUSDT'
    interval = '1d'
    start_time = '1609459200000' # 1Jan 2021
    end_time = '1648416000000' # 27 March 2023
    limit = '500'

    # Получение исторической информации о ценах закрытия eth и btc за последний год и формирование датафреймов
    params_eth = {'symbol': symbol_eth, 'interval': interval, 'startTime': start_time, 'endTime': end_time, 'limit': limit}
    response_eth = requests.get(endpoint, params=params_eth)
    df_eth = pd.DataFrame(response_eth.json(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df_eth['timestamp'] = pd.to_datetime(df_eth['timestamp'], unit='ms')
    df_eth.set_index('timestamp', inplace=True)
    df_eth['close'] = df_eth['close'].astype(float)

    params_btc = {'symbol': symbol_btc, 'interval': interval, 'startTime': start_time, 'endTime': end_time, 'limit': limit}
    response_btc = requests.get(endpoint, params=params_btc)
    df_btc = pd.DataFrame(response_btc.json(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
    df_btc.set_index('timestamp', inplace=True)
    df_btc['close'] = df_btc['close'].astype(float)

    # рассчет изменения цен токенов в процентах
    eth_daily_pct_change = df_eth['close'].pct_change().dropna()
    btc_daily_pct_change = df_btc['close'].pct_change().dropna()

    df_combined = pd.concat([eth_daily_pct_change, btc_daily_pct_change], axis=1).dropna()

    # Определение зависимой и независимой переменной (eth, btc)
    y = df_combined.iloc[:, 0]
    X = df_combined.iloc[:, 1].values.reshape(-1, 1)

    reg = LinearRegression().fit(X, y)
    print("Coefficients: ", reg.coef_)

    # Вывод квадратичного значения
    print("R-squared: ", reg.score(X, y))
    return reg.score(X, y)


# Получение актуальной информации о цене eth
async def get_eth():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            eth_price = float(data['price'])
            return eth_price

'''Получаем изменение цены eth в процентах, затем умножаем на квадратичное значение, которое свидетельствует 
о влиянии цены btc на цену eth'''
async def check_eth_price(r_square):
    eth_prices = pd.Series()
    while True:
        eth_price = await get_eth()
        eth_prices = eth_prices.append(pd.Series(eth_price, index=[pd.Timestamp.now()]))
        eth_prices = eth_prices[eth_prices.index > pd.Timestamp.now() - pd.Timedelta(minutes=60)]
        price_change = (eth_prices.iloc[-1] - eth_prices.iloc[0]) / eth_prices.iloc[0] * 100
        if abs(price_change) * (1 - r_square) > 1:
            print(f"Изменения цены за последний час составили: {price_change:.2f}%")
        else:
            print("Цена не изменилась более чем на 1%")
        await asyncio.sleep(3600)

async def main():
    r_square = await get_square_correlation()

    await asyncio.gather(check_eth_price(r_square))

if __name__ == '__main__':
    asyncio.run(main())