import pandas as pd
import requests
from datetime import datetime, timedelta
from pandas import ExcelWriter
import xmltodict as xmltodict
from pprint import pprint


def currency_today():
    # Делаем запрос к Центробанку
    url = 'http://www.cbr.ru/scripts/XML_daily.asp?date_req='
    date = datetime.now()
    date = date.strftime('%d/%m/%Y')

    response = requests.get(url + date)
    tree = xmltodict.parse(response.content)
    date = datetime.strptime(date, '%d/%m/%Y')
    # Создаем таблицу с нужными данными
    df = pd.DataFrame(tree['ValCurs']['Valute'])
    df['Date'] = tree['ValCurs']['@Date']
    df = df.set_index(['@ID'])
    return df.loc[['R01815']], df.loc[['R01370']]


# Скачиваем файл с данными
df_VON = pd.read_excel('currency.xlsx   ', sheet_name='VON', index_col='@ID')
df_SOM = pd.read_excel('currency.xlsx', sheet_name='SOM',index_col='@ID')
VON_today, SOM_today = currency_today()
df_VON = pd.concat([df_VON,VON_today])
df_SOM = pd.concat([df_SOM,SOM_today])
df_VON = df_VON.drop_duplicates()
df_SOM = df_SOM.drop_duplicates()

print(df_VON, df_SOM)
# "Записываем в Excel"
with ExcelWriter('currency.xlsx') as writer:
    df_VON.to_excel(writer, sheet_name='VON')
    df_SOM.to_excel(writer, sheet_name='SOM')

# ttp://www.cbr.ru/scripts/XML_daily.asp?date_req=15/12/2022
