import pandas as pd
import matplotlib.pyplot as plt

df_met = pd.read_parquet("Data-clean/polarstern_weather_clean.parquet")
df_met = df_met.loc["2019-10-04 01:41:00":"2020-10-01 22:59:00"]

print(df_met.true_wind_velocity)

plt.figure()
plt.plot(df_met.true_wind_velocity)
plt.xlabel("DateTime")
plt.ylabel(r"Wind Velocity \[$m.s^2$\]")
plt.grid()
plt.show()