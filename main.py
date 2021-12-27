import time

import matplotlib.pyplot as plt
from binance import Client
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

client = Client('zCVkQpBCxx9emHAjkhFXQRsz6PGdWPwLL9zT4UfkdL5D6HTXpiWaOFZULfqaci9M'
                , 'xsP3n0q5GPvsL2N0WcdeSlznkTs8iQkNydusMKCrquSIlI2aXet9cKwyYvAFNk1R')
print(client.get_asset_balance(asset='ZIL'))

klines = client.get_historical_klines("ZILETH", Client.KLINE_INTERVAL_1MINUTE, "30 minutes ago GMT")

prices = [float(x[4]) for x in klines]
times = [x[6] for x in klines]

print(prices)
print(times)
plt.plot(times, prices)
plt.show()
# t_end = time.time() + 60 * 15
# t_end = time.time() + 10
# while time.time() < t_end:
#     zil_eth_price = client.get_symbol_ticker(symbol="ZILETH")['price']
#     current_time = time.time()
#     prices.append(zil_eth_price)
#     times.append(current_time)

times = [[int(x)] for x in times]
prices = [float(x) for x in prices]

data_X_train = times[:int(len(times) * 0.8)]
data_X_test = times[int(len(times) * 0.8):]

data_Y_train = prices[:int(len(times) * 0.8)]
data_Y_test = prices[int(len(times) * 0.8):]

model = linear_model.LinearRegression()
model.fit(data_X_train, data_Y_train)
price_prediction = model.predict(data_X_test)
# The coefficients
print("Coefficients: \n", model.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(data_Y_test, price_prediction))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(data_Y_test, price_prediction))

plt.scatter(data_X_test, data_Y_test, color="black")
plt.plot(data_X_test, price_prediction, color="blue")
plt.show()
