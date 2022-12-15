from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
import xmltodict as xmltodict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.ensemble import RandomForestRegressor

#%matplotlib inline

currences=['KRW','KGS','USD','CNY']

def currency_today(valute):
    # Делаем запрос к Центробанку
    url = 'http://www.cbr.ru/scripts/XML_daily.asp?date_req='
    date = datetime.now()
    date = date.strftime('%d/%m/%Y')
    response = requests.get(url + date)
    tree = xmltodict.parse(response.content)
    date = datetime.strptime(date, '%d/%m/%Y')
    #pprint(tree['ValCurs']['Valute'])
    # Создаем таблицу с нужными данными
    df = pd.DataFrame(tree['ValCurs']['Valute'])
    df['Date'] = tree['ValCurs']['@Date']
    df = df.set_index(['CharCode'])
    return df.loc[valute]

def convert_to_datetime(row):
    return datetime.strptime(row['Date'], '%d.%m.%Y')


#print(currency_today('USD'))
# Скачиваем файл с данными
def storage(name):
    global full
    cur_name = name
    file = name+'.xlsx'
    name = pd.read_excel(file, index_col='CharCode')
    name=name.append(currency_today(cur_name))
    #name = pd.concat([name, currency_today(cur_name)])
    name = name.drop_duplicates()
    name.to_excel(file)
    name['Value'] = name['Value'].str.replace(',', '.')
    name.Value = name.Value.apply(lambda x: float(x))
    name['Date'] = name.apply(convert_to_datetime, axis=1)
    name.pop('NumCode')
    name.pop('@ID')
    name.pop('Name')
    name.pop('Nominal')
    #name['month'] = name["Date"].dt.month

    return name

for currency in currences:
    storage(currency)

#Составляем таблицу с исходными данными
USD = storage('USD')
KRW = storage('KRW')
KGS = storage('KGS')
CNY = storage('CNY')
full= pd.merge(USD, KRW, on='Date', suffixes=('_USD', '_KRW'))
full= pd.merge(full, KGS, on='Date', suffixes=(False, '_KGS'))
full= pd.merge(full, CNY, on='Date', suffixes=(False, '_CNY'))
full = full.set_index('Date')
full.columns = ['USD', 'KRW', 'KGS', 'CNY']
full = full.drop_duplicates()
full['USD mean7days'] = full.USD.shift(1).rolling(window=7).mean()
full['KRW mean7days'] = full.KRW.shift(1).rolling(window=7).mean()
full['KGS mean7days'] = full.KGS.shift(1).rolling(window=7).mean()
full['CNY mean7days'] = full.CNY.shift(1).rolling(window=7).mean()
full['USD std_7days'] = full.USD.shift(1).rolling(window=7).std()
full['KRW std_7days'] = full.KRW.shift(1).rolling(window=7).std()
full['KGS std_7days'] = full.KGS.shift(1).rolling(window=7).std()
full['CNY std_7days'] = full.CNY.shift(1).rolling(window=7).std()
full['USD median_7days'] = full.USD.shift(1).rolling(window=7).median()
full['KRW median_7days'] = full.KRW.shift(1).rolling(window=7).median()
full['KGS median_7days'] = full.KGS.shift(1).rolling(window=7).median()
full['CNY median_7days'] = full.CNY.shift(1).rolling(window=7).median()
full['USD var_7days'] = full.USD.shift(1).rolling(window=7).var()
full['KRW var_7days'] = full.KRW.shift(1).rolling(window=7).var()
full['KGS var_7days'] = full.KGS.shift(1).rolling(window=7).var()
full['CNY var_7days'] = full.CNY.shift(1).rolling(window=7).var()
for day in range(1,8):
    for names in currences:
        name = f"{names} {day}d ago"
        full[name] = full[names].shift(day)
full = full.dropna()

plt.plot(full.USD, label="USD")
plt.plot(full.KRW, label="KRW")
plt.plot(full.KGS, label="KGS")
plt.plot(full.CNY, label="CNY")
plt.legend()
plt.show()

#Прогнозируем KGS
y = full[:-1].KGS

X = full[:-1].drop('KGS', axis=1)
#y = y.dropna()
#X= X.dropna()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train)
#print(X_train)

#print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

print("MAE = ", mean_absolute_error(y_pred, y_test))
print("MAX = ", max_error(y_pred, y_test))
print('\n')


model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MAE = ", mean_absolute_error(y_pred, y_test))
print("MAX = ", max_error(y_pred, y_test))




