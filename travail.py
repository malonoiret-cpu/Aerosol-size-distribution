import pandas as pd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ion_formation_rate3 import IonFormation as ifr
# from ion_formation_rate import IonFormation as ifr

# ---- Study settings ---------------------------------------------------------
    # Time settings -------------------------
start_w = '2019-12-01 00:00:00'
end_w = '2020-03-20 00:00:00'

npf_datetime_list_text = [['2019-12-02 14:00:00', '2019-12-06 04:00:00'],# ['2019-12-02 14:00:00', '2019-12-06 04:00:00'], # qualitatively determined blowing snow events
                     ['2019-12-16 06:00:00', '2019-12-16 15:00:00'],
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

npf_datetime_list_text = [['2019-12-02 00:00:00', '2019-12-06 00:00:00'],
						  ['2019-12-07 00:00:00', '2019-12-10 00:00:00'],
						  ['2020-01-15 00:00:00', '2020-01-16 00:00:00'],
						  ['2020-01-26 00:00:00', '2020-01-27 00:00:00'],
						  ['2020-02-02 00:00:00', '2020-02-04 00:00:00'],
						  ['2020-02-12 00:00:00', '2020-02-16 00:00:00'],
						  ['2020-02-18 00:00:00', '2020-02-22 00:00:00']]

npf_datetime_list = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in npf_datetime_list_text] # For plot_events

    # Physics settings
temperature = 298          # [K], if None, met_data considered, else considered as constant (298K was default)
pressure = 101.3             # [kPa], if None, met_data considered, else considered as constant (101.3 was default)

dia_min = .75               # diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62             # (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

roll_period = '12h'          # i.e '2h', if not None, apply a rolling median over the time given to smooth the data
diff_order = 2              # to compute dN/dt (see _diff function in the class)

    # Plot settings
bin_ranges = [(0.75,  31.62), (2.05,  2.74), (3.16, 7.5), (8.66,  31.62)]
bin_ranges0 = [(0.75,  0.75), (0.87,  0.87), (1., 1.), (1.15,  1.15)]
bin_ranges1 = [(1.33, 1.33), (1.54, 1.54), (1.78, 1.78), (2.05, 2.05)]
bin_ranges2 = [(2.37, 2.37), (2.74, 2.74), (3.16, 3.16), (3.65, 3.65)]
bin_ranges3 = [(4.22, 4.22), (4.87, 4.87), (5.62, 5.62), (6.49, 6.49)]
bin_ranges4 = [(7.5, 7.5), (8.66, 8.66), (10., 10.), (11.55, 11.55)]
bin_ranges5 = [(13.34, 13.34), (15.4, 15.4), (17.78, 17.78), (20.54, 20.54)]
bin_ranges6 = [(23.71, 23.71), (27.38, 27.38), (31.62, 31.62)]
bin_all = [bin_ranges0, bin_ranges1, bin_ranges2, bin_ranges3, bin_ranges4, bin_ranges5, bin_ranges6]

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

res_w = ifr(nais_part_pos_10min_w, nais_ion_pos_10min_w, nais_ion_neg_10min_w, met_10min_w,
			low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
			diff_order=diff_order, smooth_window=roll_period)
print("\t Results computed in the instance res_w")
# -------------------------------------------------------------------------------------------

# ---- Slice datasets on one event period and compute results -----------------------------
event_number = 1
event_dates = npf_datetime_list_text[event_number]
start_ev = event_dates[0]
end_ev = event_dates[1]

nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start_ev:end_ev]
nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start_ev:end_ev]
nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start_ev:end_ev]
met_10min = data_dic['met'].loc[start_ev:end_ev]

res = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_10min,
			low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
			diff_order=diff_order, smooth_window=roll_period)
print("The instance containing the result has been created (res)")
# ---------------------------------------------------------------------------------------

# res.plot_members(bin_ranges=bin_ranges, s = 'pos')
res_w.plot_events(s = 'pos', bin_ranges=[[0.75, 31.62]], event_list=npf_datetime_list)
plt.show()



# for bin_ran in bin_all:
# 	#res_w.plot_events(s = 'pos', bin_ranges=bin_ran, event_list=npf_datetime_list, commony=False)
# 	res_w.plot_events(s = 'neg', bin_ranges=bin_ran, event_list=npf_datetime_list, commony=False)