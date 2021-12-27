import matplotlib.pyplot as plt
import numpy as np
from binance import Client
from sklearn import linear_model

client = Client('zCVkQpBCxx9emHAjkhFXQRsz6PGdWPwLL9zT4UfkdL5D6HTXpiWaOFZULfqaci9M'
                , 'xsP3n0q5GPvsL2N0WcdeSlznkTs8iQkNydusMKCrquSIlI2aXet9cKwyYvAFNk1R')
print(client.get_asset_balance(asset='ZIL'))

klines = client.get_historical_klines("ZILETH", Client.KLINE_INTERVAL_1MINUTE, "30 minutes ago GMT")

prices = [float(x[4]) for x in klines]
times = [x[6] for x in klines]

times = [[int(x)] for x in times]
prices = [float(x) for x in prices]

data_X_train = times[:int(len(times) * 0.8)]
data_X_test = times[int(len(times) * 0.8):]

data_Y_train = prices[:int(len(times) * 0.8)]
data_Y_test = prices[int(len(times) * 0.8):]

model = linear_model.LinearRegression()
model.fit(times, prices)

print('coefficient', model.coef_)
print('intercept', model.intercept_)

x = np.linspace(times[0], times[-1])
y = (model.coef_ * x) + model.intercept_
plt.plot(times, prices, '-b')
plt.plot(x, y, '-r')
plt.show()
