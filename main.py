import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
from Model import Model
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import pickle
# from torchsummaryX import summary
import matplotlib.pyplot as plt

# choose to use gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameter settings
input_size = 6
hidden_size = 30
num_layers = 1
num_epochs = 800
learning_rate = 0.001
batch_size = 32

with open('./data/train.p', 'rb') as f:
    train = pickle.load(f)
with open('./data/test.p', 'rb') as f:
    test = pickle.load(f)

predict_days = 180

train_loader = Data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(test, batch_size=batch_size, shuffle=True)
test = torch.split(test, split_size_or_sections=[30, predict_days, predict_days, predict_days, predict_days], dim=2)

# monthly rate
# monthly_rate = pd.read_csv('./data/owlnest_monthly_rate.csv')
# monthly_rate = monthly_rate.loc[:, ['Yilan', 'Tainan', 'Nantou', 'Taitung', 'Pingtung', 'Hualien']]

# get minmax scaler
daily_rate = pd.read_csv('./data/owlnest_daily_rate.csv')
col = list(daily_rate.columns)
col[0] = 'Date'
daily_rate.columns = col
daily_rate['Date'] = daily_rate['Date'].apply(lambda x: dt.datetime.strptime(x.split(' ')[0], '%Y-%m-%d').timestamp())
daily_rate = daily_rate.loc[:, ['Date', 'Yilan', 'Tainan', 'Nantou', 'Taitung', 'Pingtung', 'Hualien']]

# minmax_scaler = MinMaxScaler(feature_range=(0, 1))
# feature_daily_rate = daily_rate[daily_rate['Date'].apply(
#     lambda x: x >= dt.datetime(2018, 4, 25).timestamp() and x < dt.datetime(2018, 5, 25).timestamp())]
# feature_daily_rate = feature_daily_rate.drop(['Date'], axis=1)
# minmax_scaler.fit(feature_daily_rate.to_numpy().reshape(-1, 1))

# define model
model = Model(input_size, hidden_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# summary(model, torch.randn(273, 6, 30).float(), torch.randn(273, 6, predict_days).float())

train_loss = []
test_loss = []
baseline_loss = []
for epoch in range(num_epochs):
    # train the model
    model.train()
    for batch, input_feature in enumerate(train_loader):
        split_dataset = torch.split(input_feature,
                                    split_size_or_sections=[30, predict_days, predict_days, predict_days, predict_days],
                                    dim=2)
        X1 = split_dataset[0].transpose(2, 1)
        X2 = split_dataset[1]
        # X3 = split_dataset[2]
        # X4 = split_dataset[3]
        Y = split_dataset[4]

        # init_hidden = model.initHidden(len(input_feature))
        optimizer.zero_grad()
        # outputs = model(X1.float(), X2.float(), init_hidden)
        outputs = model(X1.float(), X2.float())
        loss = criterion(outputs, Y.float())
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}],  Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    train_loss.append(loss.item())

    # test the model
    model.eval()
    test_X1 = test[0].transpose(2, 1)
    test_X2 = test[1]
    # test_X3 = test[2]
    # test_X4 = test[3]
    test_Y = test[4]

    test_outputs = model(test_X1.float(), test_X2.float())
    predict_loss = criterion(test_outputs, test_Y.float())
    test_loss.append(predict_loss.item())

    baseline_predict_loss = criterion(test[1], test_Y.float())
    baseline_loss.append(baseline_predict_loss)

# draw the loss of training data, testing data, and two baseline
plt.plot(train_loss, label='training loss')
plt.plot(test_loss, label='testing loss')
plt.plot(baseline_loss, label='baseline')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and Testing Loss(With Baseline)')
plt.legend(loc='best')
plt.savefig('./img/loss.png')
plt.show()

train_outputs = model(train[0].float(), train[1].float()).detach().numpy()
test_outputs = model(test[0].float(), test[1].float()).detach().numpy()
train_ans = train[4].detach().numpy()
test_ans = test[4].detach().numpy()
train_baseline = train[1].detach().numpy()
test_baseline = test[1].detach().numpy()

len_train = len(train_ans)
len_test = len(test_ans)

# recover the range of data
# for i in range(len_train):
#     train_outputs[i] = minmax_scaler.inverse_transform(train_outputs[i])
#     train_ans[i] = minmax_scaler.inverse_transform(train_ans[i])
#     train_baseline[i] = minmax_scaler.inverse_transform(train_baseline[i])
# for i in range(len_test):
#     test_outputs[i] = minmax_scaler.inverse_transform(test_outputs[i])
#     test_ans[i] = minmax_scaler.inverse_transform(test_ans[i])
#     test_baseline[i] = minmax_scaler.inverse_transform(test_baseline[i])

# with open('./data/test_output.p', 'wb') as f:
#     pickle.dump(test_outputs, f)

# 計算第n天的已知住房率和模型預測第n天住房率的loss
def BaselineLossComparison():
    baseline = []
    prediction = []
    have_known_total = []
    model_pre_total = []
    real_total = []
    for i in range(len_test):
        have_known = []
        model_pre = []
        real = []
        for j in range(6):
            have_known.append(test_baseline[i][j][predict_days - 1])
            model_pre.append(test_outputs[i][j][predict_days - 1])
            real.append(test_ans[i][j][predict_days - 1])
            have_known_total.append(test_baseline[i][j][predict_days - 1])
            model_pre_total.append(test_outputs[i][j][predict_days - 1])
            real_total.append(test_ans[i][j][predict_days - 1])
        baseline.append(torch.sqrt(criterion(torch.FloatTensor(have_known), torch.FloatTensor(real))))
        prediction.append(torch.sqrt(criterion(torch.FloatTensor(model_pre), torch.FloatTensor(real))))
    have_known_loss = torch.sqrt(criterion(torch.FloatTensor(have_known_total), torch.FloatTensor(real_total)))
    model_pre_loss = torch.sqrt(criterion(torch.FloatTensor(model_pre_total), torch.FloatTensor(real_total)))
    print('第{}天已知住房率的loss： {:.4f}'.format(predict_days, have_known_loss))
    print('模型預測第{}天住房率的loss： {:.4f}'.format(predict_days, model_pre_loss))
    return baseline, prediction


baseline, prediction = BaselineLossComparison()
x_label = ['2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04']
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.plot(baseline, label='baseline')
plt.plot(prediction, label='prediction')
plt.xlabel('預測日期')
plt.xticks(np.arange(0, len_test, 30), labels=x_label)
plt.ylabel('RMSE Loss')
plt.ylim(0, 1)
plt.title('RMSE Loss比較')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('./img/loss comparison.png')
plt.show()

all_city = ['Yilan', 'Tainan', 'Nantou', 'Taitung', 'Pingtung', 'Hualien']


def plotAnswerAndPredict(city, istrain):
    datalength = len_train if istrain else len_test
    data_ans = train_ans if istrain else test_ans
    data_pre = train_outputs if istrain else test_outputs
    folder = 'train_img' if istrain else 'test_img'
    city_idx = all_city.index(city)
    if istrain:
        x_label = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10']
    else:
        x_label = ['2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04']
    ans = []
    pre = []
    for i in range(datalength):
        ans.append(data_ans[i][city_idx][predict_days - 1])
        pre.append(data_pre[i][city_idx][predict_days - 1])

    plt.plot(ans, label='answer')
    plt.plot(pre, label='prediction')
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    if istrain:
        plt.xticks(np.arange(0, datalength, 30), labels=x_label, rotation=90)
    else:
        plt.xticks(np.arange(0, datalength, 30), labels=x_label)
    plt.xlabel('預測日期')
    plt.ylabel('預測住房率')
    plt.ylim(0, 1)
    plt.title('預測{}未來{}天住房率'.format(city, predict_days))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./img/' + folder + '/' + city + '.png')
    plt.show()


# def plotAnswerAndPredictThreeMonth(city, istrain):
#     data_ans = train_ans if istrain else test_ans
#     data_pre = train_outputs if istrain else test_outputs
#     datalength = 273 if istrain else 183
#     city_idx = all_city.index(city)
#     folder = 'train_img_three_month' if istrain else 'test_img_three_month'
#     if istrain:
#         month_days = {'1': 31, '2': 28, '3': 31, '4': 30, '5': 31, '6': 30, '7': 31, '8': 31, '9': 30}
#         x_label = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09',
#                    '2019-10']
#     else:
#         month_days = {'10': 31, '11': 30, '12': 31, '1': 31, '2': 29, '3': 31}
#         x_label = ['2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04']
#     ans = dict()
#     pre = dict()
#     for i in range(predict_days):
#         element_ans = []
#         element_pre = []
#         for j in range(datalength):
#             element_ans.append(data_ans[j][city_idx][i])
#             element_pre.append(data_pre[j][city_idx][i])
#         ans[i] = element_ans
#         pre[i] = element_pre
#
#     for i in range(predict_days):
#         curr_day = 0
#         all_month = list(month_days.keys())
#         for j in range(int(len(all_month) / 3)):
#             days = month_days[all_month[j * 3]] + month_days[all_month[j * 3 + 1]] + month_days[all_month[j * 3 + 2]]
#             plt.plot(ans[i][curr_day:curr_day + days], label='answer')
#             plt.plot(pre[i][curr_day:curr_day + days], label='prediction')
#             plt.xticks(np.arange(0, days, 30),
#                        pd.date_range(start=x_label[j * 3], end=x_label[j * 3 + 3], freq='30D').date)
#             plt.xlabel('date')
#             plt.ylabel('living rate')
#             plt.ylim(0, 0.7)
#             plt.title('Daily Rate Prediction of {}(The {} Day)'.format(city, i + 1))
#             plt.legend(loc='best')
#             plt.savefig('./img/' + folder + '/' + city + '_' + str(i + 1) + '-' + str(j + 1))
#             plt.show()
#             curr_day += days
#
#
# def plotMonthAnswerAndPredict(city, istrain):
#     if istrain:
#         month_days = {'1': 31, '2': 28, '3': 31, '4': 30, '5': 31, '6': 30, '7': 31, '8': 31, '9': 30}
#         x_label = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09']
#         ans = monthly_rate[city].tolist()[:9]
#         data_output = train_outputs
#     else:
#         month_days = {'10': 31, '11': 30, '12': 31, '1': 31, '2': 29, '3': 31}
#         x_label = ['2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03']
#         ans = monthly_rate[city].tolist()[9:]
#         data_output = test_outputs
#     folder = 'train_img_month' if istrain else 'test_img_month'
#     city_idx = all_city.index(city)
#
#     average_predict = []
#     for i in range(predict_days):
#         element_predict = []
#         curr_day = 0
#         for key, value in month_days.items():
#             month_element = data_output[curr_day: value + curr_day]
#             average_arr = []
#             for x in range(len(month_element)):
#                 average_arr.append(month_element[x][city_idx][i])
#             element_predict.append(np.mean(average_arr))
#             curr_day += value
#         average_predict.append(element_predict)
#
#     for i in range(predict_days):
#         plt.figure(figsize=(9, 7))
#         plt.plot(ans, label='answer')
#         plt.plot(average_predict[i], label='predict')
#         plt.xticks(range(len(month_days)), labels=x_label)
#         plt.xlabel('month')
#         plt.ylabel('monthly rate')
#         plt.ylim(0, 0.7)
#         plt.title('Monthly Rate of {}(The {} Day)'.format(city, i + 1))
#         plt.legend(loc='best')
#         plt.savefig('./img/' + folder + '/' + city + '_' + str(i + 1) + '.png')
#         plt.show()


for i in range(6):
    plotAnswerAndPredict(all_city[i], True)
    plotAnswerAndPredict(all_city[i], False)
    # plotAnswerAndPredictThreeMonth(all_city[i], True)
    # plotAnswerAndPredictThreeMonth(all_city[i], False)
    # plotMonthAnswerAndPredict(all_city[i], True)
    # plotMonthAnswerAndPredict(all_city[i], False)
