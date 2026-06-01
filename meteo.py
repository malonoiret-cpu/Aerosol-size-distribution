import pandas as pd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ion_formation_rate3 import IonFormation as ifr
# from ion_formation_rate import IonFormation as ifr

# ---- Study settings ---------------------------------------------------------
    # Time settings -------------------------
start_w = '2020-06-01 00:00:00'
end_w = '2020-09-01 00:00:00'

npf_datetime_list_text = [['2020-07-03 00:00:00', '2020-07-04 06:00:00'],   # qualitatively determined blowing snow events
						  ['2020-07-11 12:00:00', '2020-07-12 16:00:00']
                        ]

npf_datetime_list = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in npf_datetime_list_text] # For plot_events

    # Physics settings
temperature = None          # [K], if None, met_data considered, else considered as constant (298K was default)
pressure = None             # [kPa], if None, met_data considered, else considered as constant (101.3 was default)

dia_min = .75               # diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62             # (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

roll_period = None          # i.e '2h', if not None, apply a rolling median over the time given to smooth the data
diff_order = 2              # to compute dN/dt (see _diff function in the class)

    # Plot settings
bin_ranges = [(0.75,  1.54), (2.05,  2.74), (3.16, 7.5), (8.66,  31.62)]
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

# ---- clean result folder --------------------
results_dir = "Results"
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)

# ---- compute results for each npf event -------------
res_dic = {} # Not used so far
for start, end in npf_datetime_list_text:

    # Slice datasets over a blowing snow event
    nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start:end]
    nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start:end]
    nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start:end]
    met_10min = data_dic['met'].loc[start:end]

    res = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_df = met_10min,
              low_dia = dia_min, high_dia = dia_max, temperature=temperature, pressure=pressure,
              diff_order=diff_order, smooth_window=roll_period)

    event_name = start + 'to' + end
    res_dic[event_name] = res

    # ---- make the directory to the dedicated folder
    event_slug = f"{start[:10]}_to_{end[:10]}"
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

    # plots for each size bins
    event_dir_pb = os.path.join(event_dir, "per_bin") # Create a dedicated foler for per bin results
    os.makedirs(event_dir_pb)
    for bins in bin_all:
         filename_pos = f"{bins[0][0]}_to_{bins[-1][-1]}_pos.png"
         filename_neg = f"{bins[0][0]}_to_{bins[-1][-1]}_neg.png"

         res.plot_members(bin_ranges=bins, s = 'pos', commony = sharey, logsc = ylogscale)
         plt.savefig(os.path.join(event_dir_pb, filename_pos), dpi = qual, bbox_inches = 'tight')
         plt.close()

         res.plot_members(bin_ranges=bins, s = 'neg', commony = sharey, logsc = ylogscale)
         plt.savefig(os.path.join(event_dir_pb, filename_neg), dpi = qual, bbox_inches = 'tight')
         plt.close()


    print(f"{event_dir} done")

print(f"All event results are saved in {results_dir} in their dedicated folder")

# ---- Plot the conc and wind over the whole time window to see the events -----------
print(f"Computing the results from {start_w} to {end_w} (global period) to show events...")

nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start_w:end_w]
nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start_w:end_w]
nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start_w:end_w]
met_10min = data_dic['met'].loc[start_w:end_w]
print("\t Data loaded, computing the results...")

res_w = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_10min,
            low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
            diff_order=diff_order, smooth_window=roll_period)
print("\t Results computed in the instance res_w")

print("Saving the plots...")
res_w.plot_events(s='pos', bin_ranges= bin_ranges, event_list= npf_datetime_list, commony=sharey)
plt.savefig(os.path.join(results_dir, "pos-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
plt.close()

res_w.plot_events(s='neg', bin_ranges= bin_ranges, event_list= npf_datetime_list, commony=sharey)
plt.savefig(os.path.join(results_dir, "neg-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
plt.close()
print(f"Global period plot are saved in {results_dir}")
# ------------------------------------------------------------------------------------