# csv pre-processing
"""
Since the protocol on cleaning the csv data files is the same regardless the settings, this code pre-process the csv files to avoid doing it at every computations of exe.py
Cleaning means set the DateTime index, and convert the column headers to numeric for the psds. The clean dfs are saved as parquet files to conserve these changes.
Clean parquet files are created holding the suffix "_clean"
Clean and resampled (to have the same index beetwin the files) are created holding the suffix "_clean_10min"
"""

import pandas as pd
import os

# ---- Smoothing parameters -------
resample_time = '10min'

cwd = os.path.dirname(os.path.realpath(__file__)) # cwd of this python script
os.chdir(cwd)

print("File treatment (can take a while)")

def load_psd(filepath, divisor):
	"""Routine to load the nais et smps files.
	The divisor is used to convert units from dndlogdp to concentration"""
	df = pd.read_csv(filepath, parse_dates=['time'], index_col='time')
	df.columns = pd.to_numeric(df.columns)
	df = df / divisor
	return df.sort_index()

#%% file path dictionaries
RAW_FILES = {'smps'					:	('Data/smps_psd_5min_raw.csv', 64),
			 'nais_part_pos_file'	:	('Data/nais_pos_particles_raw.csv', 16),
			 'nais_ion_neg_file'	:	('Data/nais_neg_ions_raw.csv', 16),
			 'nais_ion_pos_file'	:	('Data/nais_pos_ions_raw.csv', 16)
			 }

CLEAN_FILES_raw = {'smps'				:	'Data-clean/smps_psd_clean.parquet',
			 'nais_part_pos_file'	:	'Data-clean/nais_pos_particles_clean.parquet',
			 'nais_ion_neg_file'	:	'Data-clean/nais_neg_ions_clean.parquet',
			 'nais_ion_pos_file'	:	'Data-clean/nais_pos_ions_clean.parquet'
			 }

CLEAN_FILES = {'smps'				:	'Data-clean/smps_psd_10min_clean.parquet',
			 'nais_part_pos_file'	:	'Data-clean/nais_pos_particles_clean_10min.parquet',
			 'nais_ion_neg_file'	:	'Data-clean/nais_neg_ions_clean_10min.parquet',
			 'nais_ion_pos_file'	:	'Data-clean/nais_pos_ions_clean_10min.parquet'
			 }

#%% checking that the paths exist
all_paths = [path for path, _ in RAW_FILES.values()] + ['Data/polarstern_weather.csv']
for filename in all_paths:
	if not os.path.exists(filename):
		raise FileNotFoundError(f"File missing: {filename}")

# load the data
psds = {name : load_psd(path, div) for name, (path, div) in RAW_FILES.items()}		# nais and smps files

met = pd.read_csv('Data/polarstern_weather.csv', index_col='date_time',				# met file
                  encoding='latin-1', low_memory=False,
                  na_values=['mm/dd/yyyy hh:mm'])
met.index = pd.to_datetime(met.index, format='%m/%d/%Y %H:%M', errors='coerce')
met = met[met.index.notna()].apply(pd.to_numeric, errors='coerce').sort_index()


os.makedirs("Data-clean", exist_ok=True)	# Create the Data-clean folder if non-existing

# creating the new clean parquets
for name, path in CLEAN_FILES.items():							# Clean and resample
	psds_10min = psds[name].resample(resample_time).median()
	psds_10min.to_parquet(path)
	print(f"{name}_10min done")

for name, path in CLEAN_FILES_raw.items():						# Clean
	psds[name].to_parquet(path)
	print(f"{name} done")

met.to_parquet('Data-clean/polarstern_weather_clean.parquet')
met_res = met.resample(resample_time).median()
met_res.to_parquet('Data-clean/polarstern_weather_clean_10min.parquet')

# Process to read and treat Matthew's notes. Done it once and it's enough, I saved a clean csv in data, with the added 'Pollution' column
df_events = pd.read_csv('Data/days_of_interest_notes_20240214.csv', header=2, sep = ';')
df_events['start [mm/dd/yy hh:mm]'] = pd.to_datetime(df_events['start [mm/dd/yy hh:mm]'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
df_events['end [mm/dd/yy hh:mm]'] = pd.to_datetime(df_events['end [mm/dd/yy hh:mm]'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
df_events = df_events.loc[df_events['Event Type'] == 'BLOWING SNOW', ['start [mm/dd/yy hh:mm]', 'end [mm/dd/yy hh:mm]', 'Event Type', 'notes']]
df_events.columns = ['start', 'end', 'Event Type', 'notes']
df_events['Pollution'] = [True, False, False, False,  True, False, False,  True, False,  True,  True, True, True,  True , True, False, False,  True, False,
							 False, False, False, False,  True, False,  True,  True,  True,  True,  True, False,  True,  True,  True, False,  True, False, False, False]

df_events.to_csv('Data/days-of-interest.csv', sep = ';')

data_dir = os.path.join(cwd, 'Data-clean')
print(f"All files have succesfully been treated. The clean CSVs are in {data_dir}")