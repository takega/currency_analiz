from datetime import datetime, timedelta

from settings import api_token
import time
import telebot
import schedule
import matplotlib.pyplot as plt
import pandas as pd
import requests
import xmltodict as xmltodict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.ensemble import RandomForestRegressor

# %matplotlib inline

currences = ['KRW', 'KGS', 'USD', 'CNY']

bot = telebot.TeleBot(api_token)



keyboard1 = telebot.types.ReplyKeyboardMarkup()
keyboard1.add('Да', 'Нет')
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Начнем работу?',  reply_markup=keyboard1)
    bot.register_next_step_handler(message, send_text)

@bot.message_handler(content_types=['text'])
def send_text(message):

    if message.text == 'Да':

        bot.send_message(message.chat.id, 'Уверены? Нажмите Да еще раз..')
        bot.register_next_step_handler(message, job)
    if message.text == 'Нет':
        bot.send_message(message.chat.id, 'Пока..')


def currency_today(valute):
    # Делаем запрос к Центробанку
    url = 'http://www.cbr.ru/scripts/XML_daily.asp?date_req='
    date = datetime.now()
    date = date.strftime('%d/%m/%Y')
    response = requests.get(url + date)
    tree = xmltodict.parse(response.content)
    date = datetime.strptime(date, '%d/%m/%Y')
    # pprint(tree['ValCurs']['Valute'])
    df = pd.DataFrame(tree['ValCurs']['Valute'])
    df['Date'] = tree['ValCurs']['@Date']
    df = df.set_index(['CharCode'])
    return df.loc[valute]


def convert_to_datetime(row):
    return datetime.strptime(row['Date'], '%d.%m.%Y')

def convert_to_val(row):
    return row['Value'] / int(row['Nominal'])


# Скачиваем файл с данными
def storage(name):
    global full
    cur_name = name
    file = name + '.xlsx'
    name = pd.read_excel(file, index_col='CharCode')
    print(f'Новая копия файла {file} создана в', datetime.now().strftime('%H:%M:%S'))
    name = name.append(currency_today(cur_name))
    # name = pd.concat([name, currency_today(cur_name)])
    name = name.drop_duplicates()
    name.to_excel(file)
    name['Value'] = name['Value'].str.replace(',', '.')
    name.Value = name.Value.apply(lambda x: float(x))
    name['Date'] = name.apply(convert_to_datetime, axis=1)
    name['Value'] = name.apply(convert_to_val, axis=1)
    name.pop('NumCode')
    name.pop('@ID')
    name.pop('Name')
    name.pop('Nominal')
    # name['month'] = name["Date"].dt.month
    return name


# Составляем таблицу с исходными данными
USD = storage('USD')
KRW = storage('KRW')
KGS = storage('KGS')
CNY = storage('CNY')
full = pd.merge(USD, KRW, on='Date', suffixes=('_USD', '_KRW'))
full = pd.merge(full, KGS, on='Date', suffixes=(None, '_KGS'))
full = pd.merge(full, CNY, on='Date', suffixes=(None, '_CNY'))
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

for day in range(1, 8):
    for names in currences:
        name = f"{names} {day}d ago"
        full[name] = full[names].shift(day)

full = full.dropna()

# Прогнозируем
def train_data(currency_name):
    full["target"] = full[currency_name].shift(-1)
    X = full[:-1].drop("target", axis=1)
    y = full[:-1].target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, y_train, X_test, y_test, X, y


def model_1(X_train,
            y_train,
            X_test,
            y_test,
            X,
            y):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Модель 1')
    print("MAE = ", mean_absolute_error(y_pred, y_test))
    print("MAX = ", max_error(y_pred, y_test))
    return model.predict(full[1:].drop("target", axis=1))


def model_2(X_train,
            y_train,
            X_test,
            y_test,
            X,
            y):
    model = RandomForestRegressor(n_estimators=1000, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Модель 2')
    print("MAE = ", mean_absolute_error(y_pred, y_test))
    print("MAX = ", max_error(y_pred, y_test))
    print('\n')
    return model.predict(full[1:].drop("target", axis=1))


def pred(message,currency_name):
    global full
    print('\n')
    X_train, y_train, X_test, y_test, X, y = train_data(currency_name)
    # pred_1 = model_1(X_train, y_train, X_test, y_test, X, y)
    pred_2 = model_2(X_train, y_train, X_test, y_test, X, y)
    full = full[1:]
    # full['Предсказание на завтра #1'] = pred_1
    full['Предсказание на завтра'] = pred_2
    result = full[[currency_name, 'Предсказание на завтра']]
    result.to_excel(currency_name + '_pred.xlsx')
    last_10 = min(result[currency_name][-10:])
    plt.plot(result[currency_name], label=currency_name)
    plt.plot(result['Предсказание на завтра'], label='предсказание')
    plt.legend()
    graf_name = currency_name + '_fig.png'
    plt.savefig(graf_name)

    bot.send_message(message.chat.id, '************')
    #bot.send_message(message.chat.id, f'Код валюты: {currency_name}')
    try:
        time_now = (datetime.now()).strftime('%Y-%m-%d')
        print(f'Курс {currency_name} сегодня ({time_now}): \n {result.loc[time_now][currency_name]}')
        bot.send_message(message.chat.id, f'Курс {currency_name} сегодня ({time_now}): \n {result.loc[time_now][currency_name]}')
    except:
        time_now = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        print(f'Курс {currency_name} сегодня ({time_now}): {result.loc[time_now][currency_name]}')
        bot.send_message(message.chat.id,
                         f'Курс {currency_name} сегодня ({time_now}):  {result.loc[time_now][currency_name]}')
    print(f'Минимальный курс {currency_name} за последние пару недель: {last_10} дата: {result.index[result[currency_name] == last_10][0]}')
    bot.send_message(message.chat.id,
                     f'Минимальный курс {currency_name} за последние пару недель: {last_10} дата: {result.index[result[currency_name] == last_10][0]}')
    #print(f'дней назад: {day_ago}')
    #bot.send_message(message.chat.id, f'дней назад: {day_ago}')
    print(f'Прогнозирую курс {currency_name} на завтра: {round(pred_2[-1], 3)}')
    bot.send_message(message.chat.id,
                     f'Прогнозирую курс {currency_name} на завтра: {round(pred_2[-1], 3)}')
    if result[currency_name][-15:].mean() > result.loc[time_now][currency_name]:
        print('Сегодня выгодный курс, по крайней мере выгоднее чем вчера')
        bot.send_message(message.chat.id,
                         'ПОКУПАЙ!!!! Сегодня весьма выгодный курс.')
    else:
        print('Сегодня не рекомендую покупать валюту')
        bot.send_message(message.chat.id,
                         f'Сегодня не рекомендую покупать {currency_name}')

    photo = open(graf_name, 'rb')
    bot.send_photo(message.chat.id, photo)


    # print(result.loc[time_now][currency_name])


def job(message):
    for currency in currences:
        storage(currency)
        pred(message, currency)
def every_day_job():
    for currency in currences:
        storage(currency)
    bot.infinity_polling(interval=0, timeout=20)



schedule.every().day.at("14:30").do(every_day_job)
while True:
    schedule.run_pending()
    time.sleep(1)
