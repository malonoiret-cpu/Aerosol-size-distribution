import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from ion_formation_rate3 import IonFormation as ifr
# from ion_formation_rate import IonFormation as ifr

# ---- Study settings ---------------------------------------------------------
	# time window (from 2019-06-20 14:46:06 to 2020-10-01 17:59:36)
start = '2019-12-02 14:00:00'
end = '2019-12-06 04:00:00'

	# diameter window (from 0.75 to 31.62 [nm])
	# (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). I the bins are wanted, uncommenting the NaN filter line in the function is required)
dia_min = .75
dia_max = 31.62

roll_period = '2h'
diff_ord = 5
qual = 150
# ---------------------------------------------------------------------------

# ---- load data -------------------------------------------------------
def load_psd(filepath):
	"""Load the nais et smps files."""
	df = pd.read_parquet(filepath)
	return df

CLEAN_FILES = {'smps'				:	'Data-clean/smps_psd_clean.parquet',
			 'nais_part_pos_file'	:	'Data-clean/nais_pos_particles_clean.parquet',
			 'nais_ion_neg_file'	:	'Data-clean/nais_neg_ions_clean.parquet',
			 'nais_ion_pos_file'	:	'Data-clean/nais_pos_ions_clean.parquet',
			 'met'                  :   'Data-clean/polarstern_weather_clean.parquet'}

data_dic = {name : load_psd(filename) for name, filename in CLEAN_FILES.items()}

print("The data have been loaded \n \t Preparing the data...")
# ----------------------------------------------------------------------

#%% ---- Prepare data -----------------------------------------------------------------------
# Slice over a blowing snow event
# smps = data_dic['smps'].loc[start:end]								# Not used !!!
nais_part_pos = data_dic['nais_part_pos_file'].loc[start:end]
nais_ion_neg = data_dic['nais_ion_neg_file'].loc[start:end]
nais_ion_pos = data_dic['nais_ion_pos_file'].loc[start:end]
met = data_dic['met'].loc[start:end]

# resample the data to a common time format for merging, and apply some smoothing
# smps_10min = smps.resample('10min').median()						# Not used !!!
nais_part_pos_10min = nais_part_pos.resample('10min').median()
nais_ion_neg_10min = nais_ion_neg.resample('10min').median()
nais_ion_pos_10min = nais_ion_pos.resample('10min').median()
met_10min = met.resample('10min').median()

if roll_period != None:
	nais_part_pos_10min = nais_part_pos_10min.rolling(window = roll_period, center = True).median()
	nais_ion_neg_10min = nais_ion_neg_10min.rolling(window = roll_period, center = True).median()
	nais_ion_pos_10min = nais_ion_pos_10min.rolling(window = roll_period, center = True).median()
	met_10min = met_10min.rolling(window = roll_period, center = True).median()

print("\t \t Data have been prepared")
# -------------------------------------------------------------------------------------------

# create the instance with the desired settings
res = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_df= met_10min, low_dia=dia_min, high_dia=dia_max, diff_order=diff_ord)
print("The instance containing the result has been created (res)")

# Prepare the folder for the savings
# ---- clean result folder --------------------
results_dir = "Results"
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
    
# ---- make the directory to the dedicated folder
    event_slug = f"{start[:10]}_to_{end[:10]}"
    event_dir = os.path.join(results_dir, event_slug)
    os.makedirs(event_dir)


res.plot_members(s='pos')
plt.savefig(os.path.join(event_dir, "members_pos.png"), dpi=qual, bbox_inches='tight')
plt.close()

res.plot_members(s='neg')
plt.savefig(os.path.join(event_dir, "members_neg.png"), dpi=qual, bbox_inches='tight')
plt.close()

res.plot_hm(s='pos')
plt.savefig(os.path.join(event_dir, "heatmap_pos.png"), dpi=150, bbox_inches='tight')
plt.close()

res.plot_hm(s='neg')
plt.savefig(os.path.join(event_dir, "heatmap_neg.png"), dpi=150, bbox_inches='tight')
plt.close()

res.plot_hm_conc(s='pos')
plt.savefig(os.path.join(event_dir, "conc_hm_pos.png"), dpi=qual, bbox_inches='tight')
plt.close()

res.plot_hm_conc(s='neg')
plt.savefig(os.path.join(event_dir, "conc_hm_neg.png"), dpi=qual, bbox_inches='tight')
plt.close()

print(f"{event_dir} done")

print(f"All results are stored in {results_dir}")