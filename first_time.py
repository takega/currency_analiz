import pandas as pd
import requests
from datetime import datetime, timedelta
from pandas import ExcelWriter
import xmltodict as xmltodict
from pprint import pprint

currences=['KRW','KGS','USD','CNY']
url = 'http://www.cbr.ru/scripts/XML_daily.asp?date_req='
date = datetime.now()-timedelta(days=30)
result_df=pd.DataFrame()
for i in range(30):
    date = date.strftime('%d/%m/%Y')
    response = requests.get(url + date)
    tree = xmltodict.parse(response.content)
    date = datetime.strptime(date, '%d/%m/%Y')
    date = date + timedelta(days=1)
    #print(tree['ValCurs'])
    # Создаем таблицу с нужными данными
    df = pd.DataFrame(tree['ValCurs']['Valute'])
    df['Date'] = tree['ValCurs']['@Date']


    df = df.set_index(['CharCode'])
    result_df=result_df.append(df.loc[currences])
for name in currences:
    file = name + '.xlsx'
    result_df.loc[name].to_excel(file)



