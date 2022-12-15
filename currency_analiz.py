import pandas as pd
import json
import requests
from datetime import datetime, timedelta
from pprint import pprint
import xml.etree.ElementTree as ET

import xmltodict as xmltodict
# Делаем запрос к Центробанку
url = 'http://www.cbr.ru/scripts/XML_daily.asp?date_req='
date = datetime.now()
date= date.strftime('%d/%m/%Y')
response = requests.get(url+date)
tree = xmltodict.parse(response.content)
#Создаем таблицу с нужными данными
df= pd.DataFrame(tree['ValCurs']['Valute'])
df['Date'] = datetime.now()
df = df.set_index(['@ID'])
result_df = df.loc[['R01815','R01370']]

print(result_df)

#ttp://www.cbr.ru/scripts/XML_daily.asp?date_req=15/12/2022
