
API_URL = "https://api.binance.com"
API1_URL = "https://api1.binance.com"
API2_URL = "https://api2.binance.com"
API3_URL = "https://api3.binance.com"


class Data:
    def __init__(self):
        import requests
        try:
            response = requests.get(API_URL)
        except requests.exceptions as e:
            response = requests.get(API1_URL)
        except requests.exceptions as e:
            response = requests.get(API2_URL)
        except requests.exceptions as e:
            response = requests.get(API3_URL)



