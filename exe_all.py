import matplotlib.pyplot as plt
import pandas as pd
from ion_formation_rate3 import IonFormation as ifr

# ---- Study settings ---------------------------------------------------------
start = '2019-12-01 00:00:00'          	# time window (from 2019-06-20 14:46:06 to 2020-10-01 17:59:36) (met data start from 2019-10-04 01:41:00)
end = '2020-03-01 00:00:00'        

dia_min = .75                          	# diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62                        	# (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

wind_threshold = 12                 	# [m.s-1] wind threshold for BSE definition ??

npf_datetime_list_text = [['2019-12-02 14:00:00', '2019-12-06 04:00:00'], # qualitatively determined blowing snow events
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

npf_datetime_list = [
    (pd.to_datetime(start), pd.to_datetime(end))
    for start, end in npf_datetime_list_text
]

bin_ranges = [(0.75,  5.0), (5.0,  11.55), (11.55, 20.0), (20.0,  31.62)]
# ---------------------------------------------------------------------------

# ---- load data -------------------------------------------------------
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
# ----------------------------------------------------------------------

# ---- compute results for each npf event -------------
res_dic = {}
for start, end in npf_datetime_list_text:

    # Slice datasets over a blowing snow event and resample for common index
    nais_part_pos = data_dic['nais_part_pos_file'].loc[start:end].resample('10min').median()
    nais_ion_neg = data_dic['nais_ion_neg_file'].loc[start:end].resample('10min').median()
    nais_ion_pos = data_dic['nais_ion_pos_file'].loc[start:end].resample('10min').median()
    met = data_dic['met'].loc[start:end].resample('10min').median()

    res = ifr(nais_part_pos, nais_ion_pos, nais_ion_neg, low_dia = dia_min, high_dia = dia_max)

    event_name = start + 'to' + end
    res_dic[event_name] = res

    res.plot_members(s='pos')

plt.show()