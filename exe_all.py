# exe_all compute the results for each npf events indicated in the npf_datetime_list_text
# It stores all figures and csvs in dedicated folder, in results_all/
# WARNING: it deletes all data in Results_all at each run !!!

import matplotlib.pyplot as plt
import pandas as pd
from ion_formation_rate3 import IonFormation as ifr
import os
import shutil

# ---- Study settings ---------------------------------------------------------
    # Time settings -------------------------
start_w = '2019-10-01 00:00:00'     # winter time window
end_w = '2020-05-15 00:00:00'

# Import events from Matthew's notes
pollution_remove = False
df_events = pd.read_csv('Data/days-of-interest.csv', sep = ';')
df_events['start'] = pd.to_datetime(df_events['start'], format='ISO8601')
df_events['end'] = pd.to_datetime(df_events['end'], format='ISO8601')

if pollution_remove == False:
	df_events = df_events.loc[df_events['Pollution'] == False, :]

bse_list = df_events[['start', 'end']].values.tolist()  # Create the list with start and end times of bses
bse_datetime = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in bse_list] # For plot_events

    # Physics settings
temperature = None          # [K], if None, met_data considered, else T considered as constant (298K was default)
pressure = None             # [kPa], if None, met_data considered, else P considered as constant (101.3 was default)

dia_min = .75               # diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62             # (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

roll_period = None          # '2h', if not None, apply a rolling median over the time given to smooth the data
diff_order = 2              # to compute dN/dt (see _diff function in the class)

    # Plot settings
bins = [0.75,  0.87,   1.0,  1.15,  1.33,  1.54,  1.78,  2.05,  2.37,  2.74,
		3.16,  3.65,  4.22,  4.87,  5.62,  6.49,   7.5,  8.66,  10.0, 11.55,
		13.34,  15.4, 17.78, 20.54, 23.71, 27.38, 31.62]

bin_all = [(size, size) for size in bins]
bin_all = bin_all[0:18] + [(10., 31.62)]    # Group the bigger ones which give the same results for clearer plot

bin_ranges = [(.75,  1.54), (2.05,  2.74), (3.16, 7.5), (8.66,  31.62)]   # for grouped subplots

sharey = False       # Share y-axis when subplotting (not on heat map)
ylogscale = False   # log scale on y-axis
qual = 150          # Output plots quality
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

# ---- clean result folder --------------------
results_dir = "Results_all"
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)
# ---- compute results for each npf event -------------
res_dic = {} # Not used so far
for start, end in bse_list:

    # Slice datasets over a blowing snow event
    nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start:end]
    nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start:end]
    nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start:end]
    met_10min = data_dic['met'].loc[start:end]

    res = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_df = met_10min, df_events=df_events,
              low_dia = dia_min, high_dia = dia_max, temperature=temperature, pressure=pressure,
              diff_order=diff_order, smooth_window=roll_period)

    event_name = f"{start.date()}_to_{end.date()}"#start + 'to' + end
    res_dic[event_name] = res

    # ---- make the directory to the dedicated folder
    event_slug = f"{start.date()}_to_{end.date()}" # [:10]
    event_dir = os.path.join(results_dir, event_slug)
    os.makedirs(event_dir)

    # ---- generate the plots and save them
    res.plot_members(bin_ranges= bin_ranges, s='pos', commony= sharey, logsc = ylogscale)
    plt.savefig(os.path.join(event_dir, "members_pos.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res.plot_members(bin_ranges= bin_ranges, s='neg', commony= sharey, logsc = ylogscale)
    plt.savefig(os.path.join(event_dir, "members_neg.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res.plot_members(bin_ranges= bin_ranges, s='ratio', commony= sharey, logsc = ylogscale)
    plt.savefig(os.path.join(event_dir, "members_ratio.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res.plot_hm(s='pos')
    plt.savefig(os.path.join(event_dir, "heatmap_pos.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res.plot_hm(s='neg')
    plt.savefig(os.path.join(event_dir, "heatmap_neg.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res.plot_hm_conc(s='pos')
    plt.savefig(os.path.join(event_dir, "conc_hm_pos.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res.plot_hm_conc(s='neg')
    plt.savefig(os.path.join(event_dir, "conc_hm_neg.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res.plot_hm_conc(s='ratio')
    plt.savefig(os.path.join(event_dir, "conc_hm_ratio.png"), dpi=qual, bbox_inches='tight')
    plt.close()
	
    res.plot_members(bin_ranges=bin_all, s = 'pos', commony = True, logsc = ylogscale)
    plt.savefig(os.path.join(event_dir, "members_all_pos.png"), dpi=qual, bbox_inches='tight')
    plt.close()
	
    res.plot_members(bin_ranges=bin_all, s = 'neg', commony = True, logsc = ylogscale)
    plt.savefig(os.path.join(event_dir, "members_all_neg.png"), dpi=qual, bbox_inches='tight')
    plt.close()
	
    note = df_events.loc[df_events['start'] == start, 'notes'].values[0]
    with open(os.path.join(event_dir, "notes.txt"), 'w') as f:
        f.write(str(note))

    print(f"{event_dir} done")

print(f"All event results are saved in {results_dir} in their dedicated folder")

# ---- Plot the conc and wind over the whole time window to see the events -----------
print(f"Computing the results from {start_w} to {end_w} (global period) to show events...")

nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start_w:end_w]
nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start_w:end_w]
nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start_w:end_w]
met_10min = data_dic['met'].loc[start_w:end_w]
print("\t Data loaded, computing the results...")

res_w = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_10min, df_events=df_events,
            low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
            diff_order=diff_order, smooth_window=roll_period)
print("\t Results computed in the instance res_w")

print("Saving the plots...")
# res_w.plot_events(s='pos', bin_ranges= bin_ranges, event_list= bse_datetime, commony=sharey)
# plt.savefig(os.path.join(results_dir, "pos-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.plot_events(s='neg', bin_ranges= bin_ranges, event_list= bse_datetime, commony=sharey)
# plt.savefig(os.path.join(results_dir, "neg-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
# plt.close()

res_w.plot_events(s='pos', bin_ranges= [[dia_min,dia_max]], event_list= bse_datetime, commony=sharey)
plt.savefig(os.path.join(results_dir, "all_pos-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
plt.close()

res_w.plot_events(s='neg', bin_ranges= [[dia_min,dia_max]], event_list= bse_datetime, commony=sharey)
plt.savefig(os.path.join(results_dir, "all_neg-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
plt.close()

add_ras = False
add_poll = False
res_w.scatter_values('pos', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
plt.savefig(os.path.join(results_dir, "scatter_dtemp_pos.png"), dpi=qual, bbox_inches='tight')
plt.close()

res_w.scatter_values('neg', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
plt.savefig(os.path.join(results_dir, "scatter_dtemp_neg.png"), dpi=qual, bbox_inches='tight')
plt.close()

res_w.scatter_values('pos', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
plt.savefig(os.path.join(results_dir, "scatter_wind_pos.png"), dpi=qual, bbox_inches='tight')
plt.close()

res_w.scatter_values('neg', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
plt.savefig(os.path.join(results_dir, "scatter_wind_neg.png"), dpi=qual, bbox_inches='tight')
plt.close()

res_w.scatter_values('pos', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
plt.savefig(os.path.join(results_dir, "scatter_temperature_pos.png"), dpi=qual, bbox_inches='tight')
plt.close()

res_w.scatter_values('neg', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
plt.savefig(os.path.join(results_dir, "scatter_temperature_neg.png"), dpi=qual, bbox_inches='tight')
plt.close()

print(f"Global period plots are saved in {results_dir}")
# ------------------------------------------------------------------------------------
