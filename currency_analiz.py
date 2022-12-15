import pandas as pd
import requests
from datetime import datetime, timedelta
from pandas import ExcelWriter
import xmltodict as xmltodict
from pprint import pprint
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

#print(currency_today('USD'))
# Скачиваем файл с данными
def storage(name):
    cur_name = name
    file = name+'.xlsx'
    name = pd.read_excel(file, index_col='CharCode')
    name=name.append(currency_today(cur_name))
    #name = pd.concat([name, currency_today(cur_name)])
    name = name.drop_duplicates()
    name.to_excel(file)
    return name

for currency in currences:
    storage(currency)


print(storage('KRW'))

#print(df_VON)
#df_VON, df_SOM, df_USD, df_CNY = currency_today()
