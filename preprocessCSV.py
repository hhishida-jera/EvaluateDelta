import pandas as pd

print("データ読み込み")

#必要部分だけ残す
df=pd.read_csv(r"CSV\Load_fromSnowflake.csv")

# StartDatetime列をdatetime型に変換
df["DELIVERYDATE"] = pd.to_datetime(df["DELIVERYDATE"], errors='coerce')

idx = df.loc[:, "AREAID"]==3
df = df.loc[idx, :]
df = df.loc[:, ["DELIVERYDATE", "MW"]]

#2024FYだけ取得
start_date = pd.Timestamp('2023-03-31')
end_date = pd.Timestamp('2025-04-01')

idx1 = df.loc[:,"DELIVERYDATE"] > start_date
idx2 = df.loc[:,"DELIVERYDATE"] < end_date
idx = idx1&idx2
df = df.loc[idx, :]

#並べ直し
df = df.sort_values(by="DELIVERYDATE")