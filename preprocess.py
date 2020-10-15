import pandas as pd
import numpy as np
import torch
import datetime as dt
import pickle
import matplotlib.pyplot as plt

daily_rate = pd.read_csv('./data/owlnest_daily_rate.csv')
col = list(daily_rate.columns)
col[0] = 'Date'
daily_rate.columns = col
daily_rate['Date'] = daily_rate['Date'].apply(lambda x: dt.datetime.strptime(x.split(' ')[0], '%Y-%m-%d').timestamp())
daily_rate = daily_rate.loc[:, ['Date', 'Yilan', 'Tainan', 'Nantou', 'Taitung', 'Pingtung', 'Hualien']]

order = pd.read_csv('./data/order.csv')
order['date'] = order['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp())
order['booking_dt'] = order['booking_dt'].apply(lambda x: dt.datetime.strptime(x.split(' ')[0], '%Y-%m-%d').timestamp())

hotel = pd.read_csv('./data/hotel.csv')
hotel['start_dt'] = hotel['start_dt'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp())
hotel['end_dt'] = hotel['end_dt'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp())

vacation = pd.read_csv('./data/vacation.csv')
vacation = vacation.loc[:, ['Date', 'isVacation']]
vacation['Date'] = vacation['Date'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d').timestamp())
vacation['isVacation'] = vacation['isVacation'].apply(lambda x: 1 if x else 0)

predict_days = 7

# choose the range of training set and testing set
training_start_year = dt.datetime.fromtimestamp(dt.datetime(2019, 1, 1).timestamp() - 86400 * predict_days).year
training_start_month = dt.datetime.fromtimestamp(dt.datetime(2019, 1, 1).timestamp() - 86400 * predict_days).month
training_start_day = dt.datetime.fromtimestamp(dt.datetime(2019, 1, 1).timestamp() - 86400 * predict_days).day
middle_year = dt.datetime.fromtimestamp(dt.datetime(2019, 10, 1).timestamp() - 86400 * predict_days).year
middle_month = dt.datetime.fromtimestamp(dt.datetime(2019, 10, 1).timestamp() - 86400 * predict_days).month
middle_day = dt.datetime.fromtimestamp(dt.datetime(2019, 10, 1).timestamp() - 86400 * predict_days).day
testing_end_year = dt.datetime.fromtimestamp(dt.datetime(2019, 4, 1).timestamp() - 86400 * predict_days).year
testing_end_month = dt.datetime.fromtimestamp(dt.datetime(2019, 4, 1).timestamp() - 86400 * predict_days).month
testing_end_day = dt.datetime.fromtimestamp(dt.datetime(2019, 4, 1).timestamp() - 86400 * predict_days).day

def getFeatureTarget(year1, month1, day1, year2, month2, day2):
    feature = []
    feature_2 = []  # daily rate that is already known
    feature_3 = []  # the month of the day
    feature_4 = []  # is vacation or not
    target = []
    for row in daily_rate.itertuples():
        if row.Date >= dt.datetime(year1, month1, day1).timestamp() and row.Date < dt.datetime(year2, month2, day2).timestamp():
            # find the target daily rate
            future_date = []
            for i in range(1, predict_days + 1):
                future_date.append(row.Date + 86400 * i)
                future_date = sorted(future_date)
            future_table = daily_rate[daily_rate['Date'].isin(future_date)]
            for i in range(1, len(future_table.columns)):
                element = future_table.iloc[:, [0, i]]
                element.set_index(element.columns[0])
                element = future_table.iloc[:, [i]]
                target.append(element.to_numpy().tolist())

            # find the month of the day as feature_3
            for i in range(1, len(future_table.columns)):
                element = []
                for j in range(len(future_date)):
                    curr_month = dt.datetime.fromtimestamp(future_date[j]).month
                    element.append(curr_month)
                feature_3.append(element)

            # find whether the future day vacation or not
            for i in range(1, len(future_table.columns)):
                vacation_table = vacation[vacation['Date'].isin(future_date)]
                element = vacation_table['isVacation'].tolist()
                feature_4.append(element)

            # find the daily rate of the future days that has been already known before current date
            before_date = order[order['booking_dt'].apply(lambda x: x <= row.Date)]
            for i in range(1, len(daily_rate.columns)):
                city_before_date = before_date[before_date['city'] == daily_rate.columns[i]]
                city_element = []
                hotel_in_city = hotel[hotel['city'] == daily_rate.columns[i]]
                hotel_in_city = hotel_in_city[hotel_in_city['start_dt'].apply(lambda x: x <= row.Date)]
                hotel_in_city = hotel_in_city[hotel_in_city['end_dt'].apply(lambda x: x >= row.Date)]
                total_room = sum(hotel_in_city['total_rooms'])

                for j in range(len(future_date)):
                    city_before_date_filtering = city_before_date[city_before_date['date'] == future_date[j]]
                    not_cancel_element = city_before_date_filtering[city_before_date_filtering['status'] == 0]
                    cancel_element = city_before_date_filtering[city_before_date_filtering['status'] == 1]
                    cancel_element = cancel_element[cancel_element['cancel_dt'].apply(
                        lambda x: dt.datetime.strptime(x.split(' ')[0], '%Y-%m-%d').timestamp() > row.Date)]
                    city_before_date_filtering = pd.concat([not_cancel_element, cancel_element])
                    num = sum(city_before_date_filtering['qty'])
                    if total_room == 0:
                        element = 0.0
                    else:
                        element = num / total_room
                    city_element.append(element)
                feature_2.append(np.array(city_element))

            # find the daily rate of past 30 days
            past_date = []
            for i in range(1, 31):
                past_date.append(row.Date - 86400 * i)
                past_date = sorted(past_date)
            past_table = daily_rate[daily_rate['Date'].isin(past_date)]
            for i in range(1, len(past_table.columns)):
                element = past_table.iloc[:, [0, i]]
                element.set_index(element.columns[0])
                element = past_table.iloc[:, [i]]
                feature.append(element.to_numpy().tolist())

    feature = np.array(feature).reshape(int(len(feature) / 6), 6, 30)
    feature_2 = np.array(feature_2).reshape(int(len(feature_2) / 6), 6, predict_days)
    feature_3 = np.array(feature_3).reshape(int(len(feature_3) / 6), 6, predict_days)
    feature_4 = np.array(feature_4).reshape(int(len(feature_4) / 6), 6, predict_days)
    target = np.array(target).reshape(int(len(target) / 6), 6, predict_days)

    return feature, feature_2, feature_3, feature_4, target


# training set: 20190101~20190930
# testing set: 20191001~20200331
train_feature, train_feature_2, train_feature_3, train_feature_4, train_target =\
    getFeatureTarget(training_start_year, training_start_month, training_start_day, middle_year, middle_month, middle_day)
test_feature, test_feature_2, test_feature_3, test_feature_4, test_target =\
    getFeatureTarget(middle_year, middle_month, middle_day, testing_end_year, testing_end_month, testing_end_day)

len_train = len(train_target)
len_test = len(test_target)

x_label = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10']
for i in range(6):
    ans = []
    base = []
    city = daily_rate.columns[i + 1]
    for j in range(len_train):
        ans.append(train_target[j][i][predict_days - 1])
        base.append(train_feature_2[j][i][predict_days - 1])
    plt.plot(ans, label='answer')
    plt.plot(base, label='baseline')
    plt.xticks(np.arange(0, len_train, 30), labels=x_label, rotation=90)
    plt.xlabel('date')
    plt.ylabel('living rate')
    plt.ylim(0, 0.8)
    plt.title('Daily Rate Baseline of {}(The {} Day)'.format(city, predict_days))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./img/train_img_baseline/' + city + '.png')
    plt.show()

x_label = ['2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03']
for i in range(6):
    ans = []
    base = []
    city = daily_rate.columns[i + 1]
    for j in range(len_test):
        ans.append(test_target[j][i][predict_days - 1])
        base.append(test_feature_2[j][i][predict_days - 1])
    plt.plot(ans, label='answer')
    plt.plot(base, label='baseline')
    plt.xticks(np.arange(0, len_test, 30), labels=x_label)
    plt.xlabel('date')
    plt.ylabel('living rate')
    plt.ylim(0, 0.8)
    plt.title('Daily Rate Baseline of {}(The {} Day)'.format(city, predict_days))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./img/test_img_baseline/' + city + '.png')
    plt.show()

# min max normalization
# minmax_scaler = MinMaxScaler(feature_range=(0, 1))
# feature_daily_rate = daily_rate[daily_rate['Date'].apply(
#     lambda x: x >= dt.datetime(2018, 4, 25).timestamp() and x < dt.datetime(2018, 5, 25).timestamp())]
# feature_daily_rate = feature_daily_rate.drop(['Date'], axis=1)
# minmax_scaler.fit(feature_daily_rate.to_numpy().reshape(-1, 1))
# train_feature = minmax_scaler.transform(train_feature.reshape(-1, 1)).reshape(len_train, 6, 30)
# train_feature_2 = minmax_scaler.transform(train_feature_2.reshape(-1, 1)).reshape(len_train, 6, predict_days)
# train_target = minmax_scaler.transform(train_target.reshape(-1, 1)).reshape(len_train, 6, predict_days)
# test_feature = minmax_scaler.transform(test_feature.reshape(-1, 1)).reshape(len_test, 6, 30)
# test_feature_2 = minmax_scaler.transform(test_feature_2.reshape(-1, 1)).reshape(len_test, 6, predict_days)
# test_target = minmax_scaler.transform(test_target.reshape(-1, 1)).reshape(len_test, 6, predict_days)

# minmax_scaler_month = MinMaxScaler(feature_range=(0, 1))
# minmax_scaler_month.fit(np.array(range(1, 13)).reshape(-1, 1))
# train_feature_3 = minmax_scaler_month.transform(train_feature_3.reshape(-1, 1)).reshape(len_train, 6, predict_days)
# test_feature_3 = minmax_scaler_month.transform(test_feature_3.reshape(-1, 1)).reshape(len_test, 6, predict_days)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_feature = torch.from_numpy(train_feature).to(device)
train_feature_2 = torch.from_numpy(train_feature_2).to(device)
train_feature_3 = torch.from_numpy(train_feature_3).to(device)
train_feature_4 = torch.from_numpy(train_feature_4).to(device)
train_target = torch.from_numpy(train_target).to(device)
test_feature = torch.from_numpy(test_feature).to(device)
test_feature_2 = torch.from_numpy(test_feature_2).to(device)
test_feature_3 = torch.from_numpy(test_feature_3).to(device)
test_feature_4 = torch.from_numpy(test_feature_4).to(device)
test_target = torch.from_numpy(test_target).to(device)

train = torch.cat((train_feature, train_feature_2, train_feature_3, train_feature_4, train_target), dim=2)
test = torch.cat((test_feature, test_feature_2, test_feature_3, test_feature_4, test_target), dim=2)

with open('./data/train.p', 'wb') as f:
    pickle.dump(train, f)
with open('./data/test.p', 'wb') as f:
    pickle.dump(test, f)
