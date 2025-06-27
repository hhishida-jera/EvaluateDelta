import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

