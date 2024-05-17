import requests


apikey = 'Q9DV29LM9GZVHW7Z'
# https://www.alphavantage.co/query?function=RSI&month=2020-01&series_type=open&symbol=SPY&interval=5min&time_period=14&apikey=Q9DV29LM9GZVHW7Z&outputsize=full&extended_hours=false


def getAvailableIndicators():
    url = "https://twelve-data1.p.rapidapi.com/technical_indicators"

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)

    print(response.status_code)
    return response.json()


def getTimeSeries(symbol, interval, outputsize):
    url = "https://twelve-data1.p.rapidapi.com/time_series"

    querystring = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "format": "json"}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getTimeSeries_AV(symbol, interval, month):
    # month == YYYY-MM
    # interval - the following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY' \
          f'&month={month}&symbol={symbol}&interval={interval}&apikey={apikey}&outputsize=full'

    response = requests.get(url)

    print(response.status_code)
    return response.json()


def getATR(interval, symbol, time_period, outputsize):
    url = "https://twelve-data1.p.rapidapi.com/atr"

    querystring = {"interval": interval, "symbol": symbol, "time_period": time_period, "outputsize": outputsize,
                   "format": "json"}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getATR_AV(symbol, interval, time_period, month):

    url = f'https://www.alphavantage.co/query?function=ATR' \
          f'&month={month}&symbol={symbol}&interval={interval}&time_period={time_period}' \
          f'&apikey={apikey}&outputsize=full'

    response = requests.get(url)

    print(response.status_code)
    return response.json()


def getOBV(symbol, interval, outputsize):
    url = "https://twelve-data1.p.rapidapi.com/obv"

    querystring = {"symbol": symbol, "interval": interval, "format": "json", "outputsize": outputsize,
                   "series_type": "close"}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getOBV_AV(symbol, interval, month):

    url = f'https://www.alphavantage.co/query?function=OBV' \
          f'&symbol={symbol}&interval={interval}&month={month}&apikey={apikey}&outputsize=full'

    response = requests.get(url)

    print(response.status_code)
    return response.json()


def getRSI(symbol, interval, time_period, outputsize):
    url = "https://twelve-data1.p.rapidapi.com/rsi"

    querystring = {"interval": interval, "symbol": symbol, "format": "json", "time_period": time_period,
                   "series_type": "close",
                   "outputsize": outputsize}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getRSI_AV(symbol, interval, time_period, month, series_type='close'):

    url = f'https://www.alphavantage.co/query?function=RSI' \
          f'&month={month}&symbol={symbol}&interval={interval}&time_period={time_period}' \
          f'&series_type={series_type}&apikey={apikey}&outputsize=full'

    response = requests.get(url)

    print(response.status_code)
    return response.json()


def getMACD(symbol, interval, signal_period, outputsize, fast_period, slow_period):
    url = "https://twelve-data1.p.rapidapi.com/macd"

    querystring = {"interval": interval, "symbol": symbol, "signal_period": signal_period, "outputsize": outputsize,
                   "series_type": "close", "fast_period": fast_period, "slow_period": slow_period, "format": "json"}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getMACD_AV(symbol, interval, signal_period, month, fast_period, slow_period, series_type='open'):

    url = f'https://www.alphavantage.co/query?function=MACDEXT' \
          f'&symbol={symbol}&interval={interval}&signalperiod={signal_period}&month={month}' \
          f'&fastperiod={fast_period}&slowperiod={slow_period}&series_type={series_type}&apikey={apikey}' \
          f'&fastmatype=1&slowmatype=1&signalmatype=1&outputsize=full'

    response = requests.get(url)

    print(response.status_code)
    return response.json()
