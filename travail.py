import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ion_formation_rate3 import IonFormation3 as ifr2
# from ion_formation_rate import IonFormation as ifr

# ---- Study settings ---------------------------------------------------------
	# time window (from 2019-06-20 14:46:06 to 2020-10-01 17:59:36)
start = '2019-12-03 00:01'
end = '2019-12-06 00:00'

	# diameter window (from 0.75 to 31.62 [nm])
	# (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). I the bins are wanted, uncommenting the NaN filter line in the function is required)
dia_min = .75
dia_max = 31.62
# ---------------------------------------------------------------------------

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

# merge nais and smps to have one dataframe with all size bins
# nais_to_merge = nais_part_pos_10min.loc[:,0.75:1.54]
# smps_to_merge = smps_10min.loc[:,10.6:495.8]
# merged_particle_psd = pd.concat([nais_to_merge, smps_to_merge], axis=1)

print("\t \t Data have been prepared")
# -------------------------------------------------------------------------------------------

# create the instance with the desired settings
res = ifr2(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, low_dia=dia_min, high_dia=dia_max)
print("The instance containing the result has been created (res)")


# # print csv
# print("Number of bins : \n")
# print("nucmode_pos_ion_psd : ", len(res.nucmode_pos_ion_psd.columns), 'bins')
# print(res.nucmode_pos_ion_psd.columns, '\n')

# print("nucmode_neg_ion_psd : ", len(res.nucmode_neg_ion_psd.columns), 'bins')
# print(res.nucmode_neg_ion_psd.columns, '\n')

# print("nucmode_particle_psd : ", len(res.nucmode_particle_psd.columns), 'bins')
# print(res.nucmode_particle_psd.columns, '\n')

# # print results
# print("Q_snow_pos : ", len(res.Q_snow_pos.columns), 'bins')
# print(res.Q_snow_pos, '\n')

# print('dNdp_pos/dt')
# print(res.dNdp_pos_ion / res.dtime)

# print('pos_coag_term')
# print(res.pos_coag_loss_term, '\n')

# print('alpha term')
# print(res.pos_alpha_term, '\n')

# print('chi_term')
# print(res.pos_chi_term, '\n')


# ---- plot the results ----------------------
# res.Q_snow_plot()
res.plot_members(s='pos')
res.plot_members(s='neg')

res.plot_hm(s='pos', vmini = None, vmaxi = None, cmap = "RdBu_r")
res.plot_hm(s='neg')
plt.show()
# -------------------------------------------