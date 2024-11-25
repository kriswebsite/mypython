from binance.client import Client

API_KEY = 'ZsKcRvWjcwsuQHvmtN0EZ8SU9PQvjcmycX7Eod17RByypw5GBDbr1rvHI5WGbCv4'
API_SECRET = 'pLOrAOfuyiwdOCEbwYBGxhjjjo6Z0LiFT3mB5eiEwhsPX4YkP3XlOOqBPpRMDkyL'

client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

def check_balance():
    account = client.get_account()
    for balance in account['balances']:
        if balance['asset'] in ['DOGE', 'USDT']:
            print(f"{balance['asset']}: {balance['free']}")

check_balance()
