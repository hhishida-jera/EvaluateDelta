import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import myFunc 
import itertools

##########################################################################
# インタラクティブモード ON（interactive on）
plt.ion()  
flag_vis = False

##########################################################################
# それぞれ読み込み
print("1. データ読み込み")

print("Solar")
df_s=pd.read_csv(r"CSV\forecast_2024FY_tokyo_for_each_weather_scenario.csv  ")

# 条件に合う列インデックスを選択（4で割って余りが0または3）
cols = [i for i in range(df_s.shape[1]) if i % 4 == 0 or i % 4 == 3]
cols.insert(-1, df_s.shape[1]-2)
df_s = df_s.iloc[:, cols]
'''
plt.figure()
plt.plot(df_s.iloc[:, 0], df_s.iloc[:, 1],label="test")
ax = plt.gca()  # 現在のAxesオブジェクトを取得
ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
plt.xticks(rotation=45)  # X軸ラベルを90度回転
plt.title("Solar")
plt.legend()
plt.tight_layout()
plt.show()
'''

print("Wind")
df_w_tmp=pd.read_csv(r"CSV\Wind_FY24.csv")

# 数値列をGWhに変換し、小数点以下1桁に揃える
cols = df_w_tmp.columns[:-1]
df_w_tmp[cols] = (df_w_tmp[cols] ).round(1)
df_w_tmp[cols] = (df_w_tmp[cols] )/1000

# Solarから日付をコピー
df_w=df_s.copy()
for ii in range(1, 45):
    df_w.iloc[:, 2*ii-1] = df_w_tmp.iloc[:,ii-1].copy()
df_w.iloc[:, -1] = df_w_tmp.iloc[:, -2].copy()

'''
plt.figure()
plt.plot(df_w.iloc[:, -1], df_w.iloc[:, 0],label="test")
plt.legend()
ax = plt.gca()  # 現在のAxesオブジェクトを取得
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  
plt.xticks(rotation=45)  # X軸ラベルを90度回転
plt.title("Wind")
plt.tight_layout()
plt.show()
'''

#loadは予測と実績が別（DB問題）
print("Load")
df_l=pd.read_csv(r"CSV\Load_701.csv", parse_dates=["startDateTime"])
df_l_act=pd.read_csv(r"CSV\Load_fromSnowflake.csv", parse_dates=["DELIVERYDATE"])

# うるう年の 2月29日 データを削除
df_l = df_l[~((df_l["startDateTime"].dt.month == 2) & (df_l["startDateTime"].dt.day == 29))]
df_l_act['DELIVERYDATE'] = pd.to_datetime(df_l_act['DELIVERYDATE'])
df_l_act = df_l_act[~((df_l_act["DELIVERYDATE"].dt.month == 2) & (df_l_act["DELIVERYDATE"].dt.day == 29))]
df_l_act = df_l_act.sort_values(by='DELIVERYDATE', ascending=True)

# 不要なエリアを除外&実績のみ残す
df_l_act = df_l_act[df_l_act['DELIVERYDATE'] >= pd.Timestamp('2024-04-01')]
df_l_act = df_l_act[df_l_act['DELIVERYDATE'] < pd.Timestamp('2025-04-01')]
df_l_act = df_l_act[df_l_act["AREAID"] == 3]
df_l_act.rename(columns={'MW': 'Load_Actual_GWh'}, inplace=True)

# 空の DataFrame を用意（列を順次追加していく）、# taskId は 1〜44
result_df = pd.DataFrame()
for task_id in range(1, 45):
    task_df = df_l[df_l["taskId"] == task_id].sort_values("startDateTime")

    start_times = task_df["weatherPathDateTime"].reset_index(drop=True).copy()
    values = task_df["value"].reset_index(drop=True).copy()

    # 列名に task ID を含める（例: task01_time, task01_value）
    time_col_name = f"task{task_id:02d}_time"
    value_col_name = f"task{task_id:02d}_value"

    result_df[time_col_name] = start_times
    result_df[value_col_name] = values

#マージ & GWhへの変換
df_l = pd.concat([result_df.reset_index(drop=True), df_l_act.reset_index(drop=True)], axis=1)
for ii in range(1,45):
    v_tmp = df_l.iloc[:,2*ii-1].copy()
#    df_l.iloc[:,2*ii-1] = (v_tmp.astype(np.float64))/1000
    col_name = df_l.columns[2*ii - 1]
    df_l[col_name] = v_tmp.copy().astype(np.float64) / 1000

v_tmp = df_l.iloc[:,-1].copy()
col_name = df_l.columns[- 1]
df_l[col_name] = v_tmp.copy().astype(np.float64) / 1000

##################################################################
print("2. 集約")

print("load - wind - solar")
df_lws = df_l.copy()
for ii in range(1,45):
    vl = df_l.iloc[:,2*ii-1].copy()
    vw = df_w.iloc[:,2*ii-1].copy()
    vs = df_s.iloc[:,2*ii-1].copy()
    vl = vl-vw-vs
    col_name = df_lws.columns[2*ii - 1]
    df_lws[col_name] = vl.astype(np.float64)

vl = df_l.iloc[:,-1].copy()
vw = df_w.iloc[:,-1].copy()
vs = df_s.iloc[:,-1].copy()
vl = vl-vw-vs
col_name = df_lws.columns[- 1]
df_lws[col_name] = vl.astype(np.float64) 


print("load 日週月")
df_l_day = myFunc.MergeToPeriod(df_l, 88, "Date")
#月曜日始まり、日曜日終わり 
df_l_week = myFunc.MergeToPeriod(df_l, 88, "Week")
df_l_month = myFunc.MergeToPeriod(df_l, 88, "Month")

print("load-wind-solar 日週月")
df_lws_day = myFunc.MergeToPeriod(df_lws, 88, "Date")
#月曜日始まり、日曜日終わり 
df_lws_week = myFunc.MergeToPeriod(df_lws, 88, "Week")
df_lws_month = myFunc.MergeToPeriod(df_lws, 88, "Month")

##################################################################
print("3. 可視化")

if flag_vis == True:
    print("過去44年 折れ線表示")
    myFunc.MyBoxPlot(df_lws_month, 44, "Month", 
                    "Monthly Total (Load-Wind-Solar) against 2024FY, 1981--2024", "Monthly total [GWh]",
                r"results/MonthlyTotal_histrical_all.png")
    myFunc.MyBoxPlot(df_lws_week, 44, "Week", 
                    "Weekly Total (Load-Wind-Solar) against 2024FY, 1981--2024", "Weekly total [GWh]",
                r"results/WeeklyTotal_histrical_all.png")
    myFunc.MyBoxPlot(df_lws_day, 44, "Day", 
                    "Daily Total (Load-Wind-Solar) against 2024FY, 1981--2024", "Daily total [GWh]",
                r"results/DailyTotal_histrical_all.png")


#print("過去10年 折れ線表示")
#df_lws_month_recent=df_lws_month.iloc[:, 34:].copy()
#df_lws_week_recent=df_lws_week.iloc[:, 34:].copy()
#df_lws_day_recent=df_lws_day.iloc[:, 34:].copy()

#myFunc.MyBoxPlot(df_lws_month_recent, 10, "Month", 
#                 "Monthly Total (Load-Wind-Solar) against 2024FY, recent 10 years", "Monthly total [GWh]",
#              r"results/MonthlyTotal_histrical_recent10years.png")
#myFunc.MyBoxPlot(df_lws_week_recent, 10, "Week", 
#                 "Weekly Total (Load-Wind-Solar) against 2024FY, recent 10 years", "Weekly total [GWh]",
#              r"results/WeeklyTotal_histrical_recent10years.png")
#myFunc.MyBoxPlot(df_lws_day_recent, 10, "Day", 
#                 "Daily Total (Load-Wind-Solar) against 2024FY, recent 10 years", "Daily total [GWh]",
#              r"results/DailyTotal_histrical_recent10years.png")


##########################################################################
# 感応度評価
# （面倒なので）代表的な場所のみを選択
print("4. 感応度評価")
print("Station情報取得")
# データ読み込み
df_station = pd.read_csv(r"CSV_tokyo\Tokyo.csv", header=2)

# 列名を明示的に設定（必要に応じて確認・修正）
df_station.columns = ['timestamp', 'temperature_2m', 'relative_humidity_2m']

# 日時を datetime 型に変換
df_station['timestamp'] = pd.to_datetime(df_station['timestamp'])

# 年度情報（4月始まり）を追加
df_station['FY'] = df_station['timestamp'].apply(lambda x: x.year if x.month >= 4 else x.year - 1)

# 2/29を削除
df_station = df_station[~((df_station['timestamp'].dt.month == 2) & (df_station['timestamp'].dt.day == 29))]

# 2025FY を除外（未完了）
df_station = df_station[df_station['FY'] < 2025]

######################################
#追加
# 日付列（0列目）を datetime に変換（すでに datetime 型なら不要）
dates = pd.to_datetime(df_station.iloc[:, 0])
dates2 = pd.to_datetime(df_l.iloc[:, 0])

grad1 = []
grad2 = []

for iter in range(4,16):
    #月
    n_month = iter
    if n_month>12:
            n_month =n_month - 12
    print(f"    month {n_month}")

    # 月に該当する行のブールマスクを作成 & 2列目（インデックス1の列）を抽出
    is_month = (dates.dt.month == n_month) 
    x = df_station.iloc[is_month.values, 1]

    # 値を格納するリスト
    y_list = []
    is_month = dates2.dt.month == n_month

    # 2列ずつ（0:時刻, 1:値, 2:時刻, 3:値, ...）
    for i in range(0, 88, 2):
        # 対象の範囲
        datetime_col = pd.to_datetime(df_l.iloc[is_month.values, i])  # 時刻列（必要なら使う）
        value_col = df_l.iloc[is_month.values, i + 1]                 # 対応する値列（1, 3, ...）

        y_list.append(value_col)

    # 縦方向に連結
    y = pd.concat(y_list, ignore_index=True)

    if flag_vis == True:
        myFunc.MyScatter(x, y, 2,
                         f"Tokyo month{n_month} hourly, temperature x VS demand y", 
                         "temperature", 
                         "GWh in hour", 
                         0,
                         50,
                         fr"results/Hourly_month{n_month}_Temperature_vs_Demand.png")

    # 2列ずつ（0:時刻, 1:値, 2:時刻, 3:値, ...）
    y_list = []
    for i in range(0, 88, 2):
        # 対象の範囲
        datetime_col = pd.to_datetime(df_lws.iloc[is_month.values, i])  # 時刻列（必要なら使う）
        value_col = df_lws.iloc[is_month.values, i + 1]                 # 対応する値列（1, 3, ...）

        y_list.append(value_col)

    # 縦方向に連結
    y2 = pd.concat(y_list, ignore_index=True)

    if flag_vis == True:
        myFunc.MyScatter(x, y2, 2,
                         f"Tokyo month{n_month} hourly, temperature x VS residual demand y", 
                         "temperature", 
                         "GWh in hour", 
                         0,
                         50,
                         fr"results/Hourly_month{n_month}_Temperature_vs_ResidualDemand.png")

    #月間
    n_chunks = 44
    chunk_size = len(x) // n_chunks

    # 分割して平均
    # Series を NumPy 配列に変換して、必要な部分だけを取り出す（余りがある場合は無視）
    x_array = x.to_numpy()

    # 720個ごとに区切る（n_chunks行 × 720列 の2次元配列　）各ブロックの平均を計算（行方向に平均）
    x_reshaped = x_array.reshape(n_chunks, chunk_size)
    x_mean = x_reshaped.mean(axis=1)

    #yについては合計する
    y_array = y.to_numpy()    
    chunk_size = len(y) // n_chunks
    y_reshaped = y_array.reshape(n_chunks, chunk_size)
    y_sum = y_reshaped.sum(axis=1)

    y2_array = y2.to_numpy()
    y2_trimmed = y2_array[:n_chunks * chunk_size]
    y2_reshaped = y2_trimmed.reshape(n_chunks, chunk_size)
    y2_sum = y2_reshaped.sum(axis=1)

    #描画
    g1 = myFunc.MyScatter(x_mean, y_sum, 20,
                     f"Tokyo month{n_month} monthly, temperature x VS demand y", 
                     "temperature", 
                     "GWh in month", 
                     12000,
                     30000,
                     fr"results/Demand/Monthly_month{n_month}_Temperature_vs_Demand.png")
    grad1.append(g1)

    g2 = myFunc.MyScatter(x_mean, y2_sum, 20,
                     f"Tokyo month{n_month} monthly, temperature x VS residual demand y", 
                     "temperature", 
                     "GWh in month", 
                     12000,
                     30000,
                     fr"results/ResidualDemand/Monthly_month{n_month}_Temperature_vs_ResidualDemand.png")
    grad2.append(g2)


# リストを DataFrame に変換
df_g = pd.DataFrame(grad1, columns=['delta'])
df_g.to_csv(r'results/delta_Demand_Tokyo.csv', index=False)
df_g = pd.DataFrame(grad2, columns=['delta'])
df_g.to_csv(r'results/delta_ResidulDemand_Tokyo.csv', index=False)

