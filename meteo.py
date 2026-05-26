import pandas as pd
import matplotlib.pyplot as plt
import numpy

df_met = pd.read_parquet("Data-clean/polarstern_weather_clean.parquet")
df_met = df_met.loc["2019-10-04 01:41:00":"2020-10-01 22:59:00"]


dfs = df_met['true_wind_velocity'].resample('24h').mean()

dfs_filtered = dfs.loc[dfs >= 12]
wind_dates = dfs.index


# plt.figure()
# plt.plot(dfs)
# plt.grid()
# plt.xlabel('DateTime')
# plt.ylabel('Wind velocity [$m.s^2$]')
# plt.title('Averaged wind over 3 days')
# plt.show()