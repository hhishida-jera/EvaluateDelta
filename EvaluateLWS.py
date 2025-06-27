import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import myFunc 

# インタラクティブモード ON（interactive on）
plt.ion()  

# %%それぞれ読み込み
print("データ読み込み")
df_s=pd.read_csv(r"CSV\forecast_2024FY_tokyo_for_each_weather_scenario.csv")

# 条件に合う列インデックスを選択（4で割って余りが0または3）
cols = [i for i in range(df_s.shape[1]) if i % 4 == 0 or i % 4 == 3]

# 条件に合う列を取り出す
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
# Wind読み込み
df_w=pd.read_csv(r"CSV\Wind_FY24.csv")

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

# Load読み込み
df_l=pd.read_csv(r"CSV\Load_701.csv", parse_dates=["startDateTime"])
# うるう年の 2月29日 データを削除
df_l = df_l[~((df_l["startDateTime"].dt.month == 2) & (df_l["startDateTime"].dt.day == 29))]

# 空の DataFrame を用意（列を順次追加していく）
result_df = pd.DataFrame()

# taskId は 1〜44
for task_id in range(1, 45):
    task_df = df_l[df_l["taskId"] == task_id].sort_values("weatherPathDateTime")

    start_times = task_df["weatherPathDateTime"].reset_index(drop=True).copy()
    values = task_df["value"].reset_index(drop=True).copy()

    # 列名に task ID を含める（例: task01_time, task01_value）
    time_col_name = f"task{task_id:02d}_time"
    value_col_name = f"task{task_id:02d}_value"

    result_df[time_col_name] = start_times
    result_df[value_col_name] = values
df_l = result_df
    
'''
plt.figure()
for i in range(1,10):
    plt.plot(df_l.iloc[:, 0], df_l.iloc[:, 2*i-1],label=i)
plt.legend()
ax = plt.gca()  # 現在のAxesオブジェクトを取得
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  
plt.xticks(rotation=45)  # X軸ラベルを90度回転
plt.title("Load")
plt.tight_layout()
plt.show()
'''

# %% 
print("Load - Wind - Solarの計算")

# ダミーデータの生成（本来は df1, df2 があると仮定）
years = [f"{y}FY" for y in range(1981, 2024)]  # 1981FY〜2023FY（43列）

df_LSW = pd.DataFrame()

for ii in range(0, len(years)):
    year = years[ii]

    # 列番号でアクセスし、copy() で明示的にコピー
    vec_l = df_l.iloc[:, 2*ii - 1].copy()
    vec_s = df_s.iloc[:, 2*ii - 1].copy()
    vec_w = df_w.iloc[:, ii].copy()

    df_LSW[year] = vec_l - vec_s - vec_w

#描画
# カラーマップから色を取得
cmap = plt.colormaps['tab20'].resampled(50)
x = df_w.iloc[:, -1]  # X軸の共通データ

'''
plt.figure()
for ii in range(34, len(years)):
    year = years[ii]
    y = df_LSW.iloc[:, ii]
    color = cmap(ii)  # ii番目の色

    plt.plot(x, y,alpha=0.3, label=year)

ax = plt.gca()  # 現在のAxesオブジェクトを取得
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  
plt.xticks(rotation=45)  # X軸ラベルを90度回転
plt.title("Load - Wind - Solar, recent 10 years")
plt.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5)
)
plt.tight_layout()
plt.show()

#描画
plt.figure()
for ii in range(0, len(years)):
    year = years[ii]
    y = df_LSW.iloc[:, ii]
    color = cmap(ii)  # ii番目の色

    plt.plot(x, y,alpha=0.3, label=year)

ax = plt.gca()  # 現在のAxesオブジェクトを取得
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  
plt.xticks(rotation=45)  # X軸ラベルを90度回転
plt.title("Load - Wind - Solar, whole years")
plt.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    ncol=2
)
plt.tight_layout()
plt.show()
'''
# %% day, monthにマージ
print("day, monthにマージ")
s=20
# x を datetime に変換（念のため）
x_datetime = pd.to_datetime(x)

# ----- 日ごとの合計 -----
date_index = x_datetime.dt.floor('D')

df_daily_sum = pd.DataFrame()

for col in df_LSW.columns:
    y = df_LSW[col]
    daily_sum = y.groupby(date_index).sum()
    df_daily_sum[col] = daily_sum

df_daily_sum.index.name = 'Date'

'''
#可視化
plt.figure(figsize=(24, 12) )
#myFunc.maximize_plot_window()
for i, col in enumerate(df_daily_sum.columns):
    color = cmap(i)
    marker = 'x' 
    if i < 25:
        marker = '.' 

    if i>0:    
        plt.scatter(df_daily_sum.index, df_daily_sum[col], label=col, s=s, color=color, alpha=0.6, marker=marker)

plt.title("Daily Total, from 1981-2023")
plt.xlabel("Date")
plt.ylabel("Dairy total [kWh]")

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
plt.legend(loc='best', ncol=2)
plt.ylim(0.4e6, 1.2e6)
plt.tight_layout()
plt.grid(True)
plt.savefig(r"results/DaylyTotal_histrical.png", dpi=1200, bbox_inches='tight')
plt.show()
'''

#散布図がみづらいので、折れ線化

# 統計量の計算
mean_values = df_daily_sum.mean(axis=1)
std_values = df_daily_sum.std(axis=1)
plus_1sigma = mean_values + std_values
minus_1sigma = mean_values - std_values
max_values = df_daily_sum.max(axis=1)
min_values = df_daily_sum.min(axis=1)

'''
# プロット
plt.figure(figsize=(24, 12))

# 統計線のプロット
plt.plot(df_daily_sum.index, mean_values, color='black', label='Mean', linewidth=2)
plt.plot(df_daily_sum.index, plus_1sigma, color='blue', linestyle='--', label='+1sigma')
plt.plot(df_daily_sum.index, minus_1sigma, color='blue', linestyle='--', label='-1sigma')
plt.plot(df_daily_sum.index, max_values, color='red', linestyle=':', label='max')
plt.plot(df_daily_sum.index, min_values, color='red', linestyle=':', label='min')

# 軸と凡例の設定
plt.title("Daily Total, from 1981-2023")
plt.xlabel("Date")
plt.ylabel("Daily total [kWh]")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.legend(loc='best', ncol=2)
plt.ylim(0.4e6, 1.2e6)
plt.tight_layout()
plt.grid(True)

# 保存と表示
plt.savefig("results/DaylyTotal_histrical_plot.png", dpi=1200, bbox_inches='tight')
plt.show()
'''

# 空のdict
daily_cleaned_dict = {}
# 各列ごとにIQRで外れ値除去
for date in df_daily_sum.index:
    row = df_daily_sum.loc[date]
    q1 = row.quantile(0.25)
    q3 = row.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    cleaned_row = row.where((row >= lower) & (row <= upper))
    daily_cleaned_dict[date] = cleaned_row # Seriesやlistとして格納


# 統計量を格納する辞書
mean_values = {}
std_values = {}
plus_1sigma = {}
minus_1sigma = {}
max_values = {}
min_values = {}


# 各日付ごとに統計量を計算（NaNを除外）
for date, series in daily_cleaned_dict.items():
    series_clean = series.dropna()
    if not series_clean.empty:
        mean = series_clean.mean()
        std = series_clean.std()
        mean_values[date] = mean
        std_values[date] = std
        plus_1sigma[date] = mean + std
        minus_1sigma[date] = mean - std
        max_values[date] = series_clean.max()
        min_values[date] = series_clean.min()
    else:
        mean_values[date] = pd.NA
        std_values[date] = pd.NA
        plus_1sigma[date] = pd.NA
        minus_1sigma[date] = pd.NA
        max_values[date] = pd.NA
        min_values[date] = pd.NA

# 結果をDataFrameにまとめる
df_day_statistics = pd.DataFrame({
    'mean': pd.Series(mean_values),
    '+1sigma': pd.Series(plus_1sigma),
    '-1sigma': pd.Series(minus_1sigma),
    'max': pd.Series(max_values),
    'min': pd.Series(min_values)
})

'''
# プロット
plt.figure(figsize=(24, 12))

# 統計線のプロット

plt.plot(df_day_statistics.index, df_day_statistics['mean'], color='black', label='Mean', linewidth=2)
plt.plot(df_day_statistics.index, df_day_statistics['+1sigma'], color='blue', linestyle='--', label='+1sigma')
plt.plot(df_day_statistics.index, df_day_statistics['-1sigma'], color='blue', linestyle='--', label='-1sigma')
plt.plot(df_day_statistics.index, df_day_statistics['max'], color='red', linestyle=':', label='Max')
plt.plot(df_day_statistics.index, df_day_statistics['min'], color='red', linestyle=':', label='Min')


# 軸と凡例の設定
plt.title("Daily Total (outliner removed), from 1981-2023")
plt.xlabel("Date")
plt.ylabel("Daily total [kWh]")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.legend(loc='best', ncol=2)
plt.ylim(0.4e6, 1.2e6)
plt.tight_layout()
plt.grid(True)

# 保存と表示
plt.savefig("results/DaylyTotal_histrical_clean_plot.png", dpi=1200, bbox_inches='tight')
plt.show()
'''

# --- 外れ値除去と最大相対誤差の計算関数はそのまま ---
'''
def compute_max_relative_error(group):
    q1 = group['value'].quantile(0.25)
    q3 = group['value'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = group[(group['value'] < lower) | (group['value'] > upper)]
    outliers_all.append(outliers)

    y_prime = group[(group['value'] >= lower) & (group['value'] <= upper)]['value']
    if y_prime.empty:
        return pd.Series({'max_relative_error': np.nan})
    y_mean = y_prime.mean()
    max_error = np.max(np.abs(y_prime - y_mean)) / y_mean
    return pd.Series({'max_relative_error': max_error})
'''
# --- ここからメイン処理 ---

# day_indexはPeriodIndex（日単位）
if isinstance(x_datetime, pd.Series):
    day_index = x_datetime.dt.to_period('D')
else:
    day_index = x_datetime.to_period('D')

df_daily_sum = pd.DataFrame()
for col in df_LSW.columns:
    y = df_LSW[col]
    daily_sum = y.groupby(day_index).sum()
    df_daily_sum[col] = daily_sum

df_daily_sum.index.name = 'Day'

# PeriodIndex → Timestamp（datetime64[ns]）
x_days = df_daily_sum.index.to_timestamp()

df_box_d = df_daily_sum.copy()
df_box_d['day'] = x_days

df_melted_d = df_box_d.melt(id_vars='day', var_name='category', value_name='value')


# --- 外れ値除去と最大相対誤差の計算関数はそのまま ---
outliers_all = []
results = []

# 'day' ごとにグループ化してループ処理
for day, group in df_melted_d.groupby('day'):
    result, outliers_all = myFunc.compute_max_relative_error(group, outliers_all)
    
    # 結果が Series や dict の場合、day を追加して辞書化
    if isinstance(result, pd.Series):
        result = result.to_dict()
    result['day'] = day
    
    results.append(result)

# 結果を DataFrame に変換
error_df_d = pd.DataFrame(results).reset_index(drop=True)

#error_df_d = df_melted_d.groupby('day').apply(compute_max_relative_error).reset_index()


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(24, 12))

flierprops = dict(marker='x', color='red', markersize=6)

# 1) 箱ひげ図は day を文字列化してカテゴリ扱い
df_melted_d['day_str'] = df_melted_d['day'].dt.strftime('%Y-%m-%d')
sns.boxplot(x='day_str', y='value', data=df_melted_d, ax=ax1, flierprops=flierprops)

# 2) 折れ線グラフのxは整数インデックスに変換
unique_days = sorted(df_melted_d['day'].drop_duplicates())
day_to_int = {day: i for i, day in enumerate(unique_days)}
error_df_d['day_int'] = error_df_d['day'].map(day_to_int)

ax2.plot(error_df_d['day_int'], error_df_d['max_relative_error'], marker='o', linestyle='-', color='green')

# x軸のラベルは日付文字列を設定（90度回転）
ax2.set_xticks(range(len(unique_days)))
ax2.set_xticklabels([d.strftime('%Y-%m-%d') for d in unique_days], rotation=90)
ax2.set_xlim(-0.5, len(unique_days)-0.5)
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both', nbins=10))
ax2.set_title('Max Relative Error (Excluding Outliers)')
ax2.set_ylabel('Max |y - mean| / mean')
ax2.set_xlabel('Day')
ax2.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig(r"results/DailyTotal_histrical_boxplot.png", dpi=1200, bbox_inches='tight')
plt.show()




# %%----- 月ごとの合計 -----
# 月単位の PeriodIndex を作成（Series でも Index でも OK）
if isinstance(x_datetime, pd.Series):
    month_index = x_datetime.dt.to_period('M')
else:
    month_index = x_datetime.to_period('M')

df_monthly_sum = pd.DataFrame()

for col in df_LSW.columns:
    y = df_LSW[col]
    monthly_sum = y.groupby(month_index).sum()
    df_monthly_sum[col] = monthly_sum

#可視化準備
df_monthly_sum.index.name = 'Month'
# 月インデックス（Period）を Timestamp に変換してX軸に使う
x_months = df_monthly_sum.index.to_timestamp()

# ① box用データ整形
df_box = df_monthly_sum.copy()
df_box['month'] = x_months.strftime('%Y-%m')  # 例: '2024-08'

# ② meltで長い形式へ
df_melted = df_box.melt(id_vars='month', var_name='category', value_name='value')


# ② 外れ値除去（IQRベース）とエラー計算
outliers_all = []
results = []

# 'month' ごとにグループ化してループ処理
for month, group in df_melted.groupby('month'):
    result, outliers_all = myFunc.compute_max_relative_error(group, outliers_all)
    
    # 結果が Series や dict の場合、month を追加して辞書化
    if isinstance(result, pd.Series):
        result = result.to_dict()
    result['month'] = month
    
    results.append(result)

# 結果を DataFrame に変換
error_df = pd.DataFrame(results).reset_index(drop=True)
#error_df = df_melted.groupby('month').apply(myFunc.compute_max_relative_error).reset_index()

df_outliers = pd.concat(outliers_all, ignore_index=True)
print(df_outliers[['month', 'category', 'value']])

#可視化
plt.ioff()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(24, 12) 
#myFunc.maximize_plot_window()

# ① 箱ひげ図
# 外れ値マーカー（flier）を×印に変更
flierprops = dict(marker='x', color='red', markersize=6)
sns.boxplot(x='month', y='value', data=df_melted, ax=ax1, flierprops=flierprops)
ax1.set_title('Boxplot of Values by Month')
ax1.grid(True)

# ② 折れ線グラフ（最大の絶対偏差 / 平均）
unique_months = sorted(df_melted['month'].drop_duplicates())
month_to_int = {month: i for i, month in enumerate(unique_months)}
error_df['month_int'] = error_df['month'].map(month_to_int)

# month_intを使ってプロット
ax2.plot(error_df['month_int'], error_df['max_relative_error'], marker='o', linestyle='-', color='green')

# X軸の目盛りは整数のインデックス範囲
ax2.set_xticks(range(len(unique_months)))

# ラベルは月文字列
ax2.set_xticklabels(unique_months, rotation=90)

# X軸範囲もインデックスに合わせて指定
ax2.set_xlim(-0.5, len(unique_months) - 0.5)

# その他の設定はそのまま
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both', nbins=10))
ax2.set_ylim(0, 0.15)
ax2.set_title('Max Relative Error (Excluding Outliers)')
ax2.set_ylabel('Max |y - mean| / mean')
ax2.set_xlabel('Month')

plt.tight_layout()
plt.savefig(r"results/MonthlyTotal_histrical.png", dpi=1200, bbox_inches='tight')
plt.show()


