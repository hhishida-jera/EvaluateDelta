import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

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


def MergeToPeriod(df_l, N, PeriodType):
    result_frames = []

    # 日時列（共通）
    datetime_col = pd.to_datetime(df_l.iloc[:, 88])

    if PeriodType == "Date":
        periods = pd.date_range(start="2024-04-01", periods=365, freq="D")
        period_key = "Date"
        datetime_group = datetime_col.dt.normalize()
    elif PeriodType == "Week":
        periods = pd.period_range(start="2024-04-01", periods=53, freq="W")
        period_key = "Week"
        datetime_group = datetime_col.dt.to_period('W')
    elif PeriodType == "Month":
        periods = pd.period_range(start="2024-04", periods=12, freq="M")
        period_key = "Month"
        datetime_group = datetime_col.dt.to_period('M')
    else:
        raise ValueError("PeriodType must be one of: 'Date', 'Week', 'Month'")

    for i in range(0, N, 2):
        value_col = df_l.iloc[:, i + 1]
        year = pd.to_datetime(df_l.iloc[0, i]).year
        fy_label = f"{year}FY"

        temp_df = pd.DataFrame({
            period_key: datetime_group,
            "Value": value_col
        })

        grouped = temp_df.groupby(period_key).sum()

        # インデックス揃える
        if PeriodType == "Date":
            grouped.index = pd.to_datetime(grouped.index)
            periods_fixed = pd.to_datetime(periods)
        else:
            grouped.index = pd.PeriodIndex(grouped.index, freq=periods.freq)
            periods_fixed = periods

        grouped = grouped.reindex(periods_fixed, fill_value=0).reset_index()
        grouped = grouped.rename(columns={"Value": fy_label})

        result_frames.append(grouped[[fy_label]])

    # 横に連結
    final_df = pd.concat(result_frames, axis=1)

    # 期間列追加
    final_df[period_key] = periods_fixed

    # 🔽 最終列の "Actual_GWh" の処理を追加
    actual_values = df_l.iloc[:, -1]  # 88列目（index 88）
    actual_df = pd.DataFrame({
        period_key: datetime_group,
        "Actual_GWh": actual_values
    })
    actual_grouped = actual_df.groupby(period_key).sum()

    # 同じく reindex で整形
    if PeriodType == "Date":
        actual_grouped.index = pd.to_datetime(actual_grouped.index)
    else:
        actual_grouped.index = pd.PeriodIndex(actual_grouped.index, freq=periods.freq)

    actual_grouped = actual_grouped.reindex(periods_fixed, fill_value=0).reset_index()

    # "Actual_GWh" 列だけ追加
    final_df["Actual_GWh"] = actual_grouped["Actual_GWh"]

    return final_df

def MyScatter(x, y, s,
           titleStr, 
           XStr, 
           YStr, 
           ymin,
           ymax,
           fileStr):
    plt.figure(figsize=(24, 12))
    plt.scatter(x, y, s=s)
    plt.tight_layout()
    plt.ylabel(YStr, fontsize=18)
    plt.xlabel(XStr, fontsize=18)

    # 線形回帰の計算
    coeffs = np.polyfit(x, y, 1)  # 一次式の係数（傾きと切片）
    poly_eq = np.poly1d(coeffs)   # 回帰式の関数オブジェクト
    y_fit = poly_eq(x)            # 回帰直線のy値

    # 回帰直線の描画
    plt.plot(x, y_fit, color='red', label=f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    title_with_slope = f"{titleStr} (slope = {coeffs[0]:.2f})"
    plt.title(title_with_slope, fontsize=24)
    plt.legend(fontsize=14)

    # y軸の範囲設定（必要な場合）
    if ymin != ymax:
        plt.ylim(ymin, ymax)

    plt.grid(True)
    plt.savefig(fileStr, dpi=1200, bbox_inches='tight')
    plt.close()

    return coeffs[0]




def MyBoxPlot(df, n, DateType, titleStr, YStr, fileStr):
    fig, ax = plt.subplots(figsize=(24, 12))

    # X軸ラベル（日付）
    x_labels = df.iloc[:, -2]
    if pd.api.types.is_period_dtype(x_labels):
        x_labels = x_labels.dt.to_timestamp()
    elif not pd.api.types.is_datetime64_any_dtype(x_labels):
        x_labels = pd.to_datetime(x_labels)

    # 横位置のインデックス（整数）
    x_pos = list(range(len(df)))

    # 各行のデータをBoxPlotとして描く
    for i in x_pos:
        y_values = df.iloc[i, 0:n].values
        ax.boxplot(y_values, positions=[i], widths=0.6)

    # 折れ線（46列目 実績）を重ね書き
    line_y = df.iloc[:, -1]
    ax.plot(x_pos, line_y, marker='o', color='red', label='Actual')
    # 折れ線（44列目 2024FY 43列目 2023FY）を重ね書き
    line_y = df.iloc[:, -3]
    ax.plot(x_pos, line_y, marker='o', color='blue', label='2024FY')
    line_y = df.iloc[:, -4]
    ax.plot(x_pos, line_y, marker='o', color='black', label='2023FY')

    # X軸ラベルの間引き（DateTypeごとに処理）
    if DateType == "Month":
        tick_pos = x_pos  # 全てのラベル表示（必要なら間引き可能）
        xtick_labels = x_labels.dt.strftime('%Y-%m')
    elif DateType == "Week":
        # 9日間隔でラベルを表示（もとの設定に近い）
        tick_pos = x_pos[::9]
        xtick_labels = x_labels.dt.strftime('%Y-%m-%d')[::9]
    elif DateType == "Day":
        # 7日（1週間）間隔でラベルを表示
        tick_pos = x_pos[::7]
        xtick_labels = x_labels.dt.strftime('%Y-%m-%d')[::7]
    else:
        tick_pos = x_pos
        xtick_labels = x_labels.astype(str)

    # X軸の目盛とラベルを設定
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(xtick_labels, rotation=90)

    # タイトルなど
    plt.title(titleStr, fontsize=24)
    plt.ylabel(YStr, fontsize=18)
    plt.tight_layout()
    plt.legend(loc='best', ncol=2)
    plt.grid(True)
    plt.savefig(fileStr, dpi=1200, bbox_inches='tight')
    #plt.show()
    plt.close()
