# csv pre-processing
"""
Since the protocol on cleaning the csv data files is the same regardless the settings, this code pre-process the csv files to avoid doing it at every computations of exe.py
Clean csv files are created holding the suffix "-clean"

16/05/2026: this preprocesing is not so usefull, because most of the operations need to be made anyway in exe.py
			It could be usefull if we consider resampling, merging or other things here.
"""

import pandas as pd
import os

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

CLEAN_FILES = {'smps'				:	'Data-clean/smps_psd_5min_clean.parquet',
			 'nais_part_pos_file'	:	'Data-clean/nais_pos_particles_clean.parquet',
			 'nais_ion_neg_file'	:	'Data-clean/nais_neg_ions_clean.parquet',
			 'nais_ion_pos_file'	:	'Data-clean/nais_pos_ions_clean.parquet'
			 }

#%% checking that the paths exist
all_paths = [path for path, _ in RAW_FILES.values()] + ['Data/polarstern_weather.csv']
for filename in all_paths:
	if not os.path.exists(filename):
		raise FileNotFoundError(f"File missing: {filename}")

# load the data
psds = {name : load_psd(path, div) for name, (path, div) in RAW_FILES.items()}		# nais and smps files

met = pd.read_csv('Data/polarstern_weather.csv', index_col='date_time',				# met file (can maybe be reduced to only to columns (temp and pressure))
                  encoding='latin-1', low_memory=False,
                  na_values=['mm/dd/yyyy hh:mm'])
met.index = pd.to_datetime(met.index, format='%m/%d/%Y %H:%M', errors='coerce')
met = met[met.index.notna()].apply(pd.to_numeric, errors='coerce').sort_index()

os.makedirs("Data-clean", exist_ok=True)	# Create the Data-clean folder if non-existing
# creating the new clean parquets
for name, path in CLEAN_FILES.items():
	psds[name].to_parquet(path)
	print(f"{name} done")

met.to_parquet('Data-clean/polarstern_weather_clean.parquet')

data_dir = os.path.join(cwd, 'Data-clean')
print(f"All files have succesfully been treated. The clean CSVs are in {data_dir}")