# This code should carefully find the BSE and return it as a variable (not done yet)
# Would be great if it could return a liste of all (start, end for all events)
# since There's not so many events, the easiest is probably to set them manually.

import pandas as pd
import matplotlib.pyplot as plt
import numpy


# Set time period
start = "2019-12-01 00:00:00"
end = "2020-03-01 00:00:00"
T = '24h'                           # period for resampling (to get rid of the noise)
threshold = 1000                    # threshold for blowing snow event

def open_file(path, start=start, end=end, T = T):
    df = pd.read_parquet(path)
    df = df.loc[start:end].rolling(window = T, center = True).median()
    return df, df.sum(axis=1)

df_met = open_file("Data-clean/polarstern_weather_clean.parquet")[0] # No need for second term, the sum does not make sense
nais_part, part_all = open_file('Data-clean/nais_pos_particles_clean.parquet')
nais_ion_neg, ion_neg_all = open_file('Data-clean/nais_neg_ions_clean.parquet')
nais_ion_pos, ion_pos_all = open_file('Data-clean/nais_pos_ions_clean.parquet')


npf_datetime = ion_pos_all.loc[ion_pos_all > threshold].index

pos_ion_npf = nais_ion_pos.reindex(npf_datetime)
neg_ion_npf = nais_ion_neg.reindex(npf_datetime)

plt.figure()
# plt.plot(nais_part.sum(axis=1), label = 'N particle')
plt.plot(pos_ion_npf.sum(axis=1), '.', markersize = 0.6, color = 'tomato', label = 'N ion neg')
plt.plot(ion_pos_all, color = 'tomato', alpha = 0.5)
plt.plot(neg_ion_npf.sum(axis=1), '.', markersize = 0.6, color = 'blue', label = 'N ion pos')
plt.plot(ion_neg_all, color = 'blue', alpha = 0.5)
plt.xlabel('DateTime')
plt.ylabel('Concentration')
plt.grid()
plt.legend()
plt.show()

npf_datetime_list = [['2019-12-02 14:00:00', '2019-12-06 04:00:00'],
                     ['2019-12-15 06:00:00', '2019-12-17 12:00:00'],
                     ['2019-12-31 12:00:00', '2020-01-03 06:00:00'],
                     ['2020-01-10 00:00:00', '2020-01-18 06:00:00'],
                     ['2020-01-26 06:00:00', '2020-01-27 12:00:00'],
                     ['2020-01-29 06:00:00', '2020-01-30 18:00:00'],
                     ['2020-01-31 00:00:00', '2020-02-02 06:00:00'],
                     ['2020-02-02 06:00:00', '2020-02-04 12:00:00'],
                     ['2020-02-12 00:00:00', '2020-02-14 12:00:00'],
                     ['2020-02-18 06:00:00', '2020-02-22 18:00:00'],
                     ['2020-02-23 00:00:00', '2020-02-28 00:00:00']
                     ]

# df_met = pd.read_parquet("Data-clean/polarstern_weather_clean.parquet")
# df_met = df_met.loc[start:end]


# dfs = df_met['true_wind_velocity'].resample('24h').mean()

# dfs_filtered = dfs.loc[dfs >= 12]
# wind_dates = dfs.index

# plt.figure()
# plt.plot(dfs)
# plt.grid()
# plt.xlabel('DateTime')
# plt.ylabel('Wind velocity [$m.s^2$]')
# plt.title('Averaged wind over 3 days')
# plt.show()