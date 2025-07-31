# EvaluateDelta
## 概要
Demandモデルなどの予測値に対して気温感応度を評価します。つまり、１℃気温が変化したときに、電力需要およびRPを差し引いた電力需要が何GWhあるかを評価します。同様のことは、予測値ではなく実績値に対しても適用可能です。具体的なデータ形式は、CSVフォルダを参照ください。

## データ準備
需要のback cast予測（もしくは実績）、太陽光のback cast予測（もしくは実績）、風力のback cast予測（もしくは実績）、気温の実績値が必要です。

● 需要のback cast予測（もしくは実績）
LoadモデルをCalibration＆Simulationして得られた値を使用します。実績値を使用する場合は、JERAのDBからデータを参照・DLします。
ファイル例：Load_701.csv

● 太陽光のback cast予測（もしくは実績）
SolarモデルをCalibration＆Simulationして得られた値を使用します。実績値を使用する場合は、JERAのDBからデータを参照・DLします。
ファイル例：forecast_2024FY_tokyo_for_each_weather_scenario.csv

● 風力のback cast予測（もしくは実績）
WindモデルをCalibration＆Simulationして得られた値を使用します。ただし、2025年7月現在、モデル修復中のため、別途弊社より送付いたします。実績値を使用する場合は、JERAのDBからデータを参照・DLします。
ファイル例：Wind_FY24.csv

● 気温の実績値
気象庁のWEBサイトから、対象エリアにおける代表的な都市における、1時間解像度の気温履歴を1981/4/以降全てをDLし、１つのCSVにしておきます。
ファイル例：Tokyo.csv

## 環境準備
pyproject.tomlを参照ください。

## 実行方法
poetoryで仮想環境を構築し、適時フォルダに移動してから、例えば以下を実行ください。
>> poetry install
>> poetry shell

そのあと、EvaluateLWS.pyを実行ください。用意したファイルをそのまま使う場合以外は、気温、太陽光、風力、気温などに該当する項目を適時書き換えてください。
>> EvaluateLWS.py
