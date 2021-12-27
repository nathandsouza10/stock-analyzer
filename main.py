import time
import matplotlib.pyplot as plt
import numpy as np
from binance import Client
from sklearn import linear_model

# real_account
# client = Client('zCVkQpBCxx9emHAjkhFXQRsz6PGdWPwLL9zT4UfkdL5D6HTXpiWaOFZULfqaci9M'
#                 , 'xsP3n0q5GPvsL2N0WcdeSlznkTs8iQkNydusMKCrquSIlI2aXet9cKwyYvAFNk1R')




previous_time = time.time()

while True:
    if time.time() > previous_time + (60*5):
        # practice_account
        client = Client('3zjhWnEBC0lBlfOeYxgnrGBQlnJj3Ci7ppnQnCC1v9CGQ7HqwnQlbXlKETZAZWUC'
                        , 'XM52kHHwBiC8r88EFITS5WLVIMOOU1kHMcpLNV9EPgQHn69ctoY0uAIAWBQKmWfM')
        client.API_URL = 'https://testnet.binance.vision/api'
        print(client.get_asset_balance(asset='ETH'))
        klines = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1MINUTE, "15 minutes ago GMT")

        prices = [float(x[4]) for x in klines]
        times = [x[6] for x in klines]
        times = [[int(x)] for x in times]
        prices = [float(x) for x in prices]

        model = linear_model.LinearRegression()
        model.fit(times, prices)

        print('coefficient', model.coef_)
        print('intercept', model.intercept_)

        x = np.linspace(times[0], times[-1])
        y = (model.coef_ * x) + model.intercept_
        plt.scatter(times, prices)
        plt.plot(x, y, color='green')
        plt.show()
        previous_time = time.time()

        if model.coef_ > 0:
            print('buy eth')
        else:
            print('sell eth')
