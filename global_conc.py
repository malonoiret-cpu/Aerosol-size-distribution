import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ion_formation_rate3 import IonFormation as ifr

# ---- Study settings ---------------------------------------------------------
start = '2019-12-01 00:00:00'          # time window (from 2019-06-20 14:46:06 to 2020-10-01 17:59:36) (met data start from 2019-10-04 01:41:00)
end = '2020-03-01 00:00:00'        

dia_min = .75                       # diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62                     # (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

wind_threshold = 12                 # [m.s-1] wind threshold for BSE definition
# ---------------------------------------------------------------------------

# ---- def some functions -------------------------------------------
def wind_detect(met_df, threshold = 12):
	"""detect wind periods from met data.
	Daily average the wind velocity and keep the days over the threshold"""
	dfs = met_df['true_wind_velocity'].resample('24h').mean()
	dfs = dfs.loc[dfs >= threshold]
	return dfs

def all_plot(conc_df, met_df, wind_threshold = 12, T='72h'):
	"""Print the global concentration of particles"""
	conc_dfs = conc_df.sum(axis = 1)
	dfs_roll = conc_dfs.rolling(window=T, center = True).mean()
	wind_ev = wind_detect(met_df=met_df, threshold= wind_threshold)     #wind events
	wind_df = met_df['true_wind_velocity'].rolling(window = T, center = True).mean()
	print(wind_df)
	print(dfs_roll)
	
	fig, ax1 = plt.subplots()
	# ax1.plot(conc_dfs, '.', alpha = 0.05, color = 'blue', label = "_N")
	ax1.plot(dfs_roll, '-', color = 'blue', label = 'N')
	ax1.set_ylabel("Concentration (dN/dlogDp)", color = 'blue')
	ax1.vlines(wind_ev.index, ymin=0, ymax=np.max(conc_df), linestyles='--', color = 'red', label = "wind event")
	
	ax2 = ax1.twinx()
	ax2.plot(wind_df, '-', color = 'tomato', label = 'Daily wind')
	ax2.set_ylabel("Wind velocity ($m.s^{-1}$)", color = 'tomato')
	ax1.set_xlabel("DateTime")

	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	
	fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=1)
	plt.grid()
	plt.suptitle("Concentration over time")
	# plt.tight_layout()

	# plt.vlines(wind_df.index)

# ------------------------------------------------------------------
	
def load_psd(filepath):
	"""Load the nais et smps files."""
	df = pd.read_parquet(filepath)
	return df

CLEAN_FILES = {'smps'				:	'Data-clean/smps_psd_5min_clean.parquet',
			 'nais_part_pos_file'	:	'Data-clean/nais_pos_particles_clean.parquet',
			 'nais_ion_neg_file'	:	'Data-clean/nais_neg_ions_clean.parquet',
			 'nais_ion_pos_file'	:	'Data-clean/nais_pos_ions_clean.parquet',
			 'met'                  :   'Data-clean/polarstern_weather_clean.parquet'}

data_dic = {name : load_psd(filename) for name, filename in CLEAN_FILES.items()}

print("The data have been loaded \n \t Preparing the data...")

#%% ---- Prepare data -----------------------------------------------------------------------
# Slice over a blowing snow event
smps = data_dic['smps'].loc[start:end]								# Not used !!!
nais_part_pos = data_dic['nais_part_pos_file'].loc[start:end]
nais_ion_neg = data_dic['nais_ion_neg_file'].loc[start:end]
nais_ion_pos = data_dic['nais_ion_pos_file'].loc[start:end]
met = data_dic['met'].loc[start:end]

# resample the data to a common time format for merging, and apply some smoothing
smps_10min = smps.resample('10min').median()						# Not used !!!
nais_part_pos_10min = nais_part_pos.resample('10min').median()
nais_ion_neg_10min = nais_ion_neg.resample('10min').median()
nais_ion_pos_10min = nais_ion_pos.resample('10min').median()
met_10min = met.resample('10min').median()

print("\t \t Data have been prepared")
# -------------------------------------------------------------------------------------------

res = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, low_dia=dia_min, high_dia=dia_max)

# wind_df = wind_detect(met_10min, threshold= wind_threshold)

all_plot(res.pos_N_ion, met_df= met_10min, T = '24h')

plt.show()

