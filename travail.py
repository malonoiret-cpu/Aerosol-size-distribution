import pandas as pd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ion_formation_rate3 import IonFormation as ifr
# from ion_formation_rate import IonFormation as ifr

# ---- Study settings ---------------------------------------------------------
    # Time settings -------------------------
start_w = '2019-10-01 00:00:00'
end_w = '2020-05-15 00:00:00'

npf_datetime_list_text = [['2019-12-10 02:15:00', '2019-12-10 06:45:00'],
					['2019-12-02 14:00:00', '2019-12-06 04:00:00'],   # Qualitatively determined blowing snow events
					['2019-12-15 06:00:00', '2019-12-17 12:00:00'],
					['2019-12-31 12:00:00', '2020-01-03 06:00:00'],
					['2020-01-10 00:00:00', '2020-01-18 06:00:00'],
					['2020-01-26 06:00:00', '2020-01-27 12:00:00'],
					['2020-01-29 06:00:00', '2020-01-30 18:00:00'],
					['2020-01-31 00:00:00', '2020-02-02 06:00:00'],
					['2020-02-02 06:00:00', '2020-02-05 00:00:00'],
					['2020-02-12 00:00:00', '2020-02-14 12:00:00'],
					['2020-02-18 06:00:00', '2020-02-22 18:00:00'],
					['2020-02-23 00:00:00', '2020-02-28 00:00:00']
					]

npf_datetime_list_text1 = [['2019-12-02 00:00:00', '2019-12-06 00:00:00'],   #   Bergner et al. BSEs
						  ['2019-12-07 00:00:00', '2019-12-10 00:00:00'],
						  ['2020-01-15 00:00:00', '2020-01-16 00:00:00'],
						  ['2020-01-26 00:00:00', '2020-01-27 00:00:00'],
						  ['2020-02-02 00:00:00', '2020-02-04 00:00:00'],
						  ['2020-02-12 00:00:00', '2020-02-16 00:00:00'],
						  ['2020-02-18 00:00:00', '2020-02-22 00:00:00']]

# npf_datetime_list_text2 = [['2019-10-28 08:00:00', '2019-10-28 12:30:00'],	# Matt's events
# 						   ['2029-11-11 01:10:00', '2019-11-12 10:45:00'],
# 						   ['2019-11-15 19:15:00', '2019-11-16 01:00:00'],
# 						   ['2019-11-16 07:00:00', '2019-11-18 18:00:00'],	# polluted
# 						   ['2019-11-23 11:00:00', '2019-11-25 07:00:00'],
# 						   ['2019-12-02 15:30:00', '2019-12-06 01:00:00'],
# 						   ['2019-12-07 21:35:00', '2019-12-08 18:00:00'],	# polluted at the start of the event
# 						   ['2019-12-10 02:15:00', '2019-12-10 06:45:00'],
# 						   ['2020-01-01 08:00:00', '2020-01-03 01:00:00'],	# possibly polluted
# 						   ['2020-01-03 19:10:00', '2020-01-04 04:00:00'],	# SMPS labelled as pollution, but it is more likely blowing snow
# 						   ['2020-01-11 14:45:00', '2020-01-11 22:00:00'],	# a lot of pollution but with minimal influence?
# 						   ['2020-01-15 05:30:00', '2020-01-15 16:55:00'],	# Obvious short lived pollution spikes
# 						   ['2020-01-15 18:20:00', '2020-01-16 00:55:00'],	# Same as the previous one
# 						   ['2020-01-16 01:55:00', '2020-01-16 09:45:00'],	# Pollution flagging in the NAIS, but no evidence in the SMPS
# 						   ]

# Import events from Matthew's notes
pollution_remove = False			# if false, do not consider polluted events as events
df_events = pd.read_csv('Data/days-of-interest.csv', sep = ';')
df_events['start'] = pd.to_datetime(df_events['start'], format='%d/%m/%Y %H:%M')
df_events['end'] = pd.to_datetime(df_events['end'], format='%d/%m/%Y %H:%M')

if pollution_remove == True:
	df_events = df_events.loc[df_events['Pollution'] == False, :]

bse_list = df_events[['start', 'end']].values.tolist()
bse_datetime = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in bse_list] # For plot_events

    # Physics settings
temperature = 298          # [K], if None, met_data considered, else considered as constant (298K was default)
pressure = 101.3             # [kPa], if None, met_data considered, else considered as constant (101.3 was default)

dia_min = .75               # diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62             # (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

roll_period = None          # i.e '2h', if not None, apply a rolling median over the time given to smooth the data
diff_order = 2              # to compute dN/dt (see _diff function in the class)

    # Plot settings
def all_bin_size_by_four(bins):
	"""Make a list with all size bins suitable for plot functions"""
	bin_ranges = []
	for i in range(4, len(bins), 4):
		ranges  = [(bins[i-4], bins[i-4]), (bins[i-3], bins[i-3]), (bins[i-2], bins[i-2]), (bins[i-1], bins[i-1])]
		bin_ranges += [ranges]
	if len(bins)%4 != 0:
		nb_left_bins = len(bins)%4
		last_range = [(bins[-nb_left_bins], bins[-nb_left_bins]), (bins[-nb_left_bins+1], bins[-nb_left_bins+1]), (bins[-nb_left_bins+2], bins[-nb_left_bins+2])]
		bin_ranges += [last_range]
	return bin_ranges

def all_bin_size(bins):
	bin_ranges = []
	for size in bins:
		bin_ranges.append((size, size))
	return bin_ranges

bins = [0.75,  0.87,   1.0,  1.15,  1.33,  1.54,  1.78,  2.05,  2.37,  2.74,
		3.16,  3.65,  4.22,  4.87,  5.62,  6.49,   7.5,  8.66,  10.0, 11.55,
		13.34,  15.4, 17.78, 20.54, 23.71, 27.38, 31.62]

bin_ranges = [(0.75,  31.62), (2.05,  2.74), (3.16, 7.5), (8.66,  31.62)]   # for grouped subplots
bin_all = all_bin_size(bins)    # for unique bin subplots
bin_all = bin_all[0:18] + [(10., 31.62)]

qual = 150              # Output plots quality
sharey = False          # Share y-axis when subplotting (not on heat map)
ylogscale = False       # log scale on y-axis
qual = 150              # Output plots quality

# ---------------------------------------------------------------------------

# ---- load data -------------------------------------------------------
def load_psd(filepath):
	"""Load the nais et smps files."""
	df = pd.read_parquet(filepath)
	return df

CLEAN_FILES = {'smps'				:	'Data-clean/smps_psd_10min_clean.parquet',              # Import the resampled data
			 'nais_part_pos_file'	:	'Data-clean/nais_pos_particles_clean_10min.parquet',
			 'nais_ion_neg_file'	:	'Data-clean/nais_neg_ions_clean_10min.parquet',
			 'nais_ion_pos_file'	:	'Data-clean/nais_pos_ions_clean_10min.parquet',
			 'met'                  :   'Data-clean/polarstern_weather_clean_10min.parquet'}

data_dic = {name : load_psd(filename) for name, filename in CLEAN_FILES.items()}

print("The data have been loaded \n \t Computing the results...")
# ----------------------------------------------------------------------

#%% ---- Load datasets and comput results for the whole winter -----------------------------------------------------------------------
# Slice over the whole period considered
# smps = data_dic['smps'].loc[start:end]								# Not used !!!
nais_part_pos_10min_w = data_dic['nais_part_pos_file'].loc[start_w:end_w]
nais_ion_neg_10min_w = data_dic['nais_ion_neg_file'].loc[start_w:end_w]
nais_ion_pos_10min_w = data_dic['nais_ion_pos_file'].loc[start_w:end_w]
met_10min_w = data_dic['met'].loc[start_w:end_w]
print("\t Data loaded, computing the results...")

res_w = ifr(nais_part_pos_10min_w, nais_ion_pos_10min_w, nais_ion_neg_10min_w, met_10min_w, df_events= df_events,
			low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
			diff_order=diff_order, smooth_window=roll_period)
print("\t Results computed in the instance res_w")
# -------------------------------------------------------------------------------------------

# ---- Slice datasets on one event period and compute results -----------------------------
event_number = 35
event_dates = bse_datetime[event_number]
start_ev = event_dates[0]
end_ev = event_dates[1]

nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start_ev:end_ev]
nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start_ev:end_ev]
nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start_ev:end_ev]
met_10min = data_dic['met'].loc[start_ev:end_ev]

res = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_10min, df_events=df_events,
			low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
			diff_order=diff_order, smooth_window=roll_period)
print("The instance containing the result has been created (res)")
# ---------------------------------------------------------------------------------------


# res_w.plot_events(s='pos', bin_ranges= [[dia_min,dia_max]], event_list= bse_datetime, commony=sharey)
# res.plot_members(bin_ranges=bin_all, s= 'pos', commony=True)

# plt.show()


res_w.scatter_values('pos', x_data='wind', bin_ranges=bin_all, commony=False)
plt.show()

# wind = res_w.met_df['true_wind_velocity'].loc[start_w:end_w]
# Q_snow_pos = res_w.Q_snow_pos.loc[:,0.75:31.62].sum(axis=1)
# Q_snow_neg = res_w.Q_snow_neg.loc[:,0.75]
# wind_aligned = wind.reindex(Q_snow_pos.index)

# mask_event = res_w.event_tags == 'event'
# mask_poll = res_w.event_tags == 'event_poll'
# mask_ras = res_w.event_tags.isna()

# plt.figure()
# plt.scatter(wind_aligned[mask_ras], Q_snow_pos[mask_ras], color = 'grey', alpha=0.4, s=10, label='no event')
# # plt.scatter(wind_aligned[mask_poll], Q_snow_pos[mask_poll], color = 'tomato', alpha=0.6, s=15, label='pollution')
# plt.scatter(wind_aligned[mask_event], Q_snow_pos[mask_event], color = 'blue', alpha=1, s=15, label='event')
# plt.xlabel('Wind speed (m/s)')
# plt.ylabel('Q_snow ($cm^{-3} s{-1})')
# plt.axhline(0, color='k', lw=0.5)
# plt.legend()
# plt.show()