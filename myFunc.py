import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 図の最大化オプション関数
def maximize_plot_window():
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # Windows
    except AttributeError:
        try:
            mng.window.showMaximized()  # Linux
        except AttributeError:
            pass



# --- 外れ値除去と最大相対誤差の計算関数はそのまま ---
def compute_max_relative_error(group, outliers_all):
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
    return pd.Series({'max_relative_error': max_error}), outliers_all

def MyBoxPlot(unique_days, df_diff_day, df_daily_act, 
              str_xlabel,
              str_ylabel,
              str_title,
              str_path,
              tf_showfliers=True):
    # 各日付に対して、各年度の差分を集める
    boxplot_data = []

    for day in unique_days:
        daily_diffs = []
        for ii in range(2, 45):
            col_name = f'{ii + 1980}FY'
            if col_name in df_diff_day.columns:
                row = df_diff_day[df_diff_day["day_str"] == day]
                if not row.empty:
                    diff = (row[col_name].values[0] - df_daily_act.loc[row.index[0], 'Load_Actual_MW']) / 1000.0
                    daily_diffs.append(diff)
        boxplot_data.append(daily_diffs)

    # BoxPlotの描画
    fig, ax = plt.subplots(figsize=(24, 12))
    flierprops = dict(marker='.', markerfacecolor='black', markersize=3, linestyle='none')
    ax.boxplot(boxplot_data, positions=range(len(unique_days)), showfliers=tf_showfliers, flierprops=flierprops)


    # X軸の設定
    ax.set_xticks(range(len(unique_days)))
    ax.set_xticklabels(unique_days, rotation=90)
    ax.set_xlim(-0.5, len(unique_days) - 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

    # 軸ラベルとタイトル
    ax.set_xlabel(str_xlabel)
    ax.set_ylabel(str_ylabel)

    plt.title(str_title)

    plt.tight_layout()
    plt.savefig(str_path, dpi=1200, bbox_inches='tight')
    plt.show()

def MergeToPeriod_L(df_l, N, PeriodType):

    # 結果を格納するリスト
    result_frames = []

    # 0, 2, 4, ..., 86列目が日時、1, 3, 5, ..., 87列目が値
    for i in range(0, N, 2):
        datetime_col = df_l.iloc[:, 88]
        value_col = df_l.iloc[:, i + 1]

        # 年度名の作成（例: 2025FY）
        year = pd.to_datetime(df_l.iloc[0, i]).year
        fy_label = f"{year}FY"

        # 抽出
        if PeriodType == "Date":
            dates = pd.to_datetime(datetime_col).dt.date
            temp_df = pd.DataFrame({"Date": dates, 'Value': value_col})
            #temp_df = pd.DataFrame({'Value': value_col})
        elif PeriodType == "Month":
            months = pd.to_datetime(datetime_col).dt.to_period('M')
            temp_df = pd.DataFrame({"Month": months, 'Value': value_col})
            #temp_df = pd.DataFrame({'Value': value_col})

        # 値を合計
        grouped = temp_df.groupby(PeriodType).sum().reset_index()
        grouped = grouped.rename(columns={"Value": fy_label})

        # 結果をリストに追加
        result_frames.append(grouped[fy_label])

    # 全ての結果を横に結合（必要に応じてsuffixを付ける）
    final_df = pd.concat(result_frames, axis=1)

    return final_df
