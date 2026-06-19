# exe_all compute the results for each npf events indicated in the npf_datetime_list_text
# It stores all figures and csvs in dedicated folder, in results_all/
# WARNING: it deletes all data in Results_all at each run !!!

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ion_formation_rate3 import IonFormation as ifr
import os
import shutil

# ---- Study settings ---------------------------------------------------------
    # Time settings -------------------------
start_w = '2019-10-15 00:00:00'     # winter time window
end_w = '2020-03-18 00:00:00'

start_s = '2020-03-18 00:00:00'
end_s = '2020-10-01 00:00:00'

pollution_remove = False	# If False, polluted events are tagged as polluted event, and their results are computed. If true, their are not considered as events
spikes_remove = True		# Remove the pikes in concentration series according to the spikes_remove function

    # Physics settings
temperature = None          # [K], if None, met_data considered, else T considered as constant (298K was default)
pressure = None             # [kPa], if None, met_data considered, else P considered as constant (101.3 was default)

dia_min = .75               # diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62             # (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

roll_period = None          # '2h', if not None, apply a rolling median over the time given to smooth the data
diff_order = 2              # to compute dN/dt (see _diff function in the class)

    # Plot settings
study_poll = False			# Compute and plot 'per event' results for polluted events as well
add_ras = False 			# add non event values on scatter plots
add_poll = False			# add polluted event on scatter plots
add_npf = True              # add npf event in plots

sharey = False       # Share y-axis when subplotting (not on heat map)
ylogscale = False   # log scale on y-axis
qual = 150          # Output plots quality

bins = [0.75,  0.87,   1.0,  1.15,  1.33,  1.54,  1.78,  2.05,  2.37,  2.74,
		3.16,  3.65,  4.22,  4.87,  5.62,  6.49,   7.5,  8.66,  10.0, 11.55,
		13.34,  15.4, 17.78, 20.54, 23.71, 27.38, 31.62]

bin_all = [(size, size) for size in bins]
bin_all = bin_all[0:19] + [(11.55, 31.62)]    # Group the bigger ones which give the same results for clearer plot

bin_ranges = [(.75,  1.54), (2.05,  2.74), (3.16, 7.5), (8.66,  31.62)]   # for grouped subplots
# ---------------------------------------------------------------------------

# ---- Define events ----------------------------------------
df_events = pd.read_csv('Data/days-of-interest.csv', sep = ';')
df_events['start'] = pd.to_datetime(df_events['start'], format='ISO8601')
df_events['end'] = pd.to_datetime(df_events['end'], format='ISO8601')

# ---- load data -------------------------------------------------------
def load_psd(filepath):
	"""Load the nais et smps files."""
	df = pd.read_parquet(filepath)
	return df

def remove_spikes(df, threshold = 20000):
    """Remove values which seem to be out of range (above the threshold)"""
    mask = df.sum(axis = 1) > threshold
    df_clean = df.copy()
    df_clean.loc[mask] = np.nan
    return df_clean

CLEAN_FILES = {'smps'				:	'Data-clean/smps_psd_10min_clean.parquet',              # Import the resampled data
			 'nais_part_pos_file'	:	'Data-clean/nais_pos_particles_clean_10min.parquet',
			 'nais_ion_neg_file'	:	'Data-clean/nais_neg_ions_clean_10min.parquet',
			 'nais_ion_pos_file'	:	'Data-clean/nais_pos_ions_clean_10min.parquet',
			 'met'                  :   'Data-clean/polarstern_weather_clean_10min.parquet'}

data_dict = {name : load_psd(filename) for name, filename in CLEAN_FILES.items()}

def all_res(start, end, result_dir, data_dic, df_events):
    """Compute and save plots all results according to the settings (work with global variables)"""
    print(f"\n Analysis from {start} to {end}")
    # ---- clean result folder --------------------
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # ---- load data --------------------------------------------------
    smps_10min_win = data_dic['smps'].loc[start:end]
    nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start:end]
    nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start:end]
    nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start:end]
    met_10min = data_dic['met'].loc[start:end]

    # merge nais and smps
    bin_min_smps = dia_max + .01	# to make sure not to have twice the same column
    smps_10min_tomerge = smps_10min_win.loc[:, bin_min_smps:1000]
    nais_smps_part = pd.concat([nais_part_pos_10min, smps_10min_tomerge], axis = 1)

    if spikes_remove == True:
        nais_part_pos_10min = remove_spikes(nais_part_pos_10min, threshold = 1*10**6)
        nais_ion_neg_10min  = remove_spikes(nais_ion_neg_10min)
        nais_ion_pos_10min  = remove_spikes(nais_ion_pos_10min)

    
    res_dict = {}   # save and return all results ?
    bse_list = df_events.loc[:, ['start', 'end']].values.tolist() if study_poll else df_events.loc[df_events['Pollution'] == False, ['start', 'end']].values.tolist()

    # ---- Compute results for the global period ---------------------------------
    print(f"Computing the global period results")

    res_w = ifr(nais_smps_part, nais_ion_pos_10min, nais_ion_neg_10min, met_10min, df_events=df_events,
                low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
                diff_order=diff_order, smooth_window=roll_period)
    res_dict['global'] = res_w
    print("\t Results computed. Saving the plots...")

    res_w.plot_events(s='pos', bin_ranges= [[dia_min,dia_max]], study_poll=True, commony=sharey, T_roll='24h')
    plt.savefig(os.path.join(result_dir, "all_pos-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.plot_events(s='neg', bin_ranges= [[dia_min,dia_max]], study_poll=True, commony=sharey, T_roll='24h')
    plt.savefig(os.path.join(result_dir, "all_neg-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.scatter_values('pos', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    plt.savefig(os.path.join(result_dir, "scatter_dtemp_pos.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.scatter_values('neg', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    plt.savefig(os.path.join(result_dir, "scatter_dtemp_neg.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.scatter_values('pos', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    plt.savefig(os.path.join(result_dir, "scatter_wind_pos.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.scatter_values('neg', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    plt.savefig(os.path.join(result_dir, "scatter_wind_neg.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.scatter_values('pos', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    plt.savefig(os.path.join(result_dir, "scatter_temperature_pos.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.scatter_values('neg', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    plt.savefig(os.path.join(result_dir, "scatter_temperature_neg.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.scatter_WT('pos', bin_ranges=bin_all, commony=False, ras=add_ras, pollution=add_poll, npf=add_npf)
    plt.savefig(os.path.join(result_dir, "scatter_WT_pos.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    res_w.scatter_WT('neg', bin_ranges=bin_all, commony=False, ras=add_ras, pollution=add_poll, npf=add_npf)
    plt.savefig(os.path.join(result_dir, "scatter_WT_neg.png"), dpi=qual, bbox_inches='tight')
    plt.close()

    print(f"\t Global period plots are saved in {result_dir}")
    # ------------------------------------------------------------------------------------

    # ---- Compute results for each bse ---------------------------------------
    print(f"Computing results for each events")
    for start_ev, end_ev in bse_list:
        nais_smps_part_ev = nais_smps_part.loc[start_ev:end_ev]
        # nais_part_pos_10min_ev = nais_part_pos_10min.loc[start_ev:end_ev]
        nais_ion_neg_10min_ev = nais_ion_neg_10min.loc[start_ev:end_ev]
        nais_ion_pos_10min_ev = nais_ion_pos_10min.loc[start_ev:end_ev]
        met_10min_ev = met_10min.loc[start_ev:end_ev]
        
        res = ifr(nais_smps_part_ev, nais_ion_pos_10min_ev, nais_ion_neg_10min_ev, met_df = met_10min_ev, df_events=df_events,
                low_dia = dia_min, high_dia = dia_max, temperature=temperature, pressure=pressure,
                diff_order=diff_order, smooth_window=roll_period)
        event_name = f"{start_ev.date()}_to_{end_ev.date()}"
        res_dict[event_name] = res

        # ---- make the directory to the dedicated folder
        event_dir = os.path.join(result_dir, event_name)
        os.makedirs(event_dir)

        # ---- generate the plots and save them
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
        
        res.plot_members(bin_ranges=bin_all, s = 'pos', commony = True, logsc = ylogscale)
        plt.savefig(os.path.join(event_dir, "members_all_pos.png"), dpi=qual, bbox_inches='tight')
        plt.close()
        
        res.plot_members(bin_ranges=bin_all, s = 'neg', commony = True, logsc = ylogscale)
        plt.savefig(os.path.join(event_dir, "members_all_neg.png"), dpi=qual, bbox_inches='tight')
        plt.close()

        res.volume_plot('pos', bin_ranges=bin_all, commony=False)
        plt.savefig(os.path.join(event_dir, "volume_production_pos.png"), dpi=qual, bbox_inches='tight')
        plt.close()

        res.volume_plot('neg', bin_ranges=bin_all, commony=False)
        plt.savefig(os.path.join(event_dir, "volume_production_neg.png"), dpi=qual, bbox_inches='tight')
        plt.close()

        note = df_events.loc[df_events['start'] == start_ev, 'notes'].values[0]
        with open(os.path.join(event_dir, "notes.txt"), 'w') as f:
            f.write(str(note))
        print(f"{event_dir} done")
    print(f"All event results are saved in {result_dir} in their dedicated folder")
# ----------------------------------------------------------------------

# ---- Compute winter results ----------------------------
df_events_w = df_events.copy()
df_events_w = df_events_w[(df_events_w['start'] > pd.to_datetime(start_w)) & (df_events_w['end'] < pd.to_datetime(end_w))]

all_res(start=start_w, end = end_w, result_dir="Results_winter", data_dic=data_dict, df_events= df_events_w)


# ---- Compute summer results -----------------------
df_events_s = df_events.copy()
df_events_s = df_events_s[(df_events_s['start'] > pd.to_datetime(start_s)) & (df_events_s['end'] < pd.to_datetime(end_s))]

all_res(start=start_s, end = end_s, result_dir= "Results_summer", data_dic=data_dict, df_events= df_events_s)




# # ---- Compute winter results -------------------------
# # ---- clean result folder --------------------
# results_dir = "Results_all"
# if os.path.exists(results_dir):
#     shutil.rmtree(results_dir)
# os.makedirs(results_dir)

# # ---- load data ----------------------------------------
# nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start_w:end_w]
# nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start_w:end_w]
# nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start_w:end_w]
# met_10min = data_dic['met'].loc[start_w:end_w]

# if spikes_remove == True:
#     nais_part_pos_10min = remove_spikes(nais_part_pos_10min, threshold = 1*10**6)
#     nais_ion_neg_10min  = remove_spikes(nais_ion_neg_10min)
#     nais_ion_pos_10min  = remove_spikes(nais_ion_pos_10min)
# # ---- compute results for each npf event -------------
# res_dict = {} # Not used so far
# bse_study = bse_list if study_poll else df_events.loc[df_events['Pollution'] == False, ['start', 'end']].values.tolist()
# for start, end in bse_study:
#     # Slice datasets over a blowing snow event
#     nais_part_pos_10min_ev = nais_part_pos_10min.loc[start:end]
#     nais_ion_neg_10min_ev = nais_ion_neg_10min.loc[start:end]
#     nais_ion_pos_10min_ev = nais_ion_pos_10min.loc[start:end]
#     met_10min_ev = met_10min.loc[start:end]

#     res = ifr(nais_part_pos_10min_ev, nais_ion_pos_10min_ev, nais_ion_neg_10min_ev, met_df = met_10min_ev, df_events=df_events,
#               low_dia = dia_min, high_dia = dia_max, temperature=temperature, pressure=pressure,
#               diff_order=diff_order, smooth_window=roll_period)

#     event_name = f"{start.date()}_to_{end.date()}"
#     res_dict[event_name] = res

#     # ---- make the directory to the dedicated folder
#     event_slug = f"{start.date()}_to_{end.date()}"
#     event_dir = os.path.join(results_dir, event_slug)
#     os.makedirs(event_dir)

#     # ---- generate the plots and save them
#     res.plot_hm(s='pos')
#     plt.savefig(os.path.join(event_dir, "heatmap_pos.png"), dpi=qual, bbox_inches='tight')
#     plt.close()

#     res.plot_hm(s='neg')
#     plt.savefig(os.path.join(event_dir, "heatmap_neg.png"), dpi=qual, bbox_inches='tight')
#     plt.close()

#     res.plot_hm_conc(s='pos')
#     plt.savefig(os.path.join(event_dir, "conc_hm_pos.png"), dpi=qual, bbox_inches='tight')
#     plt.close()

#     res.plot_hm_conc(s='neg')
#     plt.savefig(os.path.join(event_dir, "conc_hm_neg.png"), dpi=qual, bbox_inches='tight')
#     plt.close()
	
#     res.plot_members(bin_ranges=bin_all, s = 'pos', commony = True, logsc = ylogscale)
#     plt.savefig(os.path.join(event_dir, "members_all_pos.png"), dpi=qual, bbox_inches='tight')
#     plt.close()
	
#     res.plot_members(bin_ranges=bin_all, s = 'neg', commony = True, logsc = ylogscale)
#     plt.savefig(os.path.join(event_dir, "members_all_neg.png"), dpi=qual, bbox_inches='tight')
#     plt.close()

#     note = df_events.loc[df_events['start'] == start, 'notes'].values[0]
#     with open(os.path.join(event_dir, "notes.txt"), 'w') as f:
#         f.write(str(note))

#     print(f"{event_dir} done")

# print(f"All event results are saved in {results_dir} in their dedicated folder")

# # ---- Plot the conc and wind over the whole time window to see the events -----------
# print(f"Computing the results from {start_w} to {end_w} (global period)")

# res_w = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_10min, df_events=df_events,
#             low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
#             diff_order=diff_order, smooth_window=roll_period)
# print("\t Results computed in the instance res_w")

# print("Saving the plots...")

# res_w.plot_events(s='pos', bin_ranges= [[dia_min,dia_max]], event_list= bse_list, commony=sharey, T_roll='24h')
# plt.savefig(os.path.join(results_dir, "all_pos-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.plot_events(s='neg', bin_ranges= [[dia_min,dia_max]], event_list= bse_list, commony=sharey, T_roll='24h')
# plt.savefig(os.path.join(results_dir, "all_neg-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.scatter_values('pos', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_dtemp_pos.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.scatter_values('neg', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_dtemp_neg.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.scatter_values('pos', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_wind_pos.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.scatter_values('neg', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_wind_neg.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.scatter_values('pos', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_temperature_pos.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.scatter_values('neg', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_temperature_neg.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.scatter_WT('pos', bin_ranges=bin_all, commony=False, ras=True, pollution=True)
# plt.savefig(os.path.join(results_dir, "scatter_WT_pos.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_w.scatter_WT('neg', bin_ranges=bin_all, commony=False, ras=True, pollution=True)
# plt.savefig(os.path.join(results_dir, "scatter_WT_neg.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# print(f"Global period plots are saved in {results_dir}")
# # ------------------------------------------------------------------------------------


# # ---- Compute "summer" results ------------------------------------------------------
# print("computing summer results")
# # ---- clean result folder --------------------
# results_dir = "Results_summer"
# if os.path.exists(results_dir):
#     shutil.rmtree(results_dir)
# os.makedirs(results_dir)

# nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start_s:end_s]
# nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start_s:end_s]
# nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start_s:end_s]
# met_10min = data_dic['met'].loc[start_s:end_s]

# # ---- compute results for each npf event -------------
# res_summer_dict = {} # Not used so far
# bse_study = bse_list_summer if study_poll else df_events_summer.loc[df_events_summer['Pollution'] == False, ['start', 'end']].values.tolist()
# print(bse_study)
# for start, end in bse_study:
#     # Slice datasets over a blowing snow event
#     nais_part_pos_10min_ev = nais_part_pos_10min.loc[start:end]
#     nais_ion_neg_10min_ev = nais_ion_neg_10min.loc[start:end]
#     nais_ion_pos_10min_ev = nais_ion_pos_10min.loc[start:end]
#     met_10min_ev = met_10min.loc[start:end]

#     res = ifr(nais_part_pos_10min_ev, nais_ion_pos_10min_ev, nais_ion_neg_10min_ev, met_df = met_10min_ev, df_events=df_events_summer,
#               low_dia = dia_min, high_dia = dia_max, temperature=temperature, pressure=pressure,
#               diff_order=diff_order, smooth_window=roll_period)

#     event_name = f"{start.date()}_to_{end.date()}"
#     res_summer_dict[event_name] = res

#     # ---- make the directory to the dedicated folder
#     event_slug = f"{start.date()}_to_{end.date()}"
#     event_dir = os.path.join(results_dir, event_slug)
#     os.makedirs(event_dir)

#     # ---- generate the plots and save them
#     res.plot_hm(s='pos')
#     plt.savefig(os.path.join(event_dir, "heatmap_pos.png"), dpi=qual, bbox_inches='tight')
#     plt.close()

#     res.plot_hm(s='neg')
#     plt.savefig(os.path.join(event_dir, "heatmap_neg.png"), dpi=qual, bbox_inches='tight')
#     plt.close()

#     res.plot_hm_conc(s='pos')
#     plt.savefig(os.path.join(event_dir, "conc_hm_pos.png"), dpi=qual, bbox_inches='tight')
#     plt.close()

#     res.plot_hm_conc(s='neg')
#     plt.savefig(os.path.join(event_dir, "conc_hm_neg.png"), dpi=qual, bbox_inches='tight')
#     plt.close()
	
#     res.plot_members(bin_ranges=bin_all, s = 'pos', commony = True, logsc = ylogscale)
#     plt.savefig(os.path.join(event_dir, "members_all_pos.png"), dpi=qual, bbox_inches='tight')
#     plt.close()
	
#     res.plot_members(bin_ranges=bin_all, s = 'neg', commony = True, logsc = ylogscale)
#     plt.savefig(os.path.join(event_dir, "members_all_neg.png"), dpi=qual, bbox_inches='tight')
#     plt.close()

#     note = df_events_summer.loc[df_events_summer['start'] == start, 'notes'].values[0]
#     with open(os.path.join(event_dir, "notes.txt"), 'w') as f:
#         f.write(str(note))

#     print(f"{event_dir} done")

#     # ---- Plot the conc and wind over the whole time window to see the events -----------
# print(f"Computing the results from {start_s} to {end_s} (global period)")

# res_s = ifr(nais_part_pos_10min, nais_ion_pos_10min, nais_ion_neg_10min, met_10min, df_events=df_events_summer,
#             low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
#             diff_order=diff_order, smooth_window=roll_period)
# print("\t Results computed in the instance res_w")

# print("Saving the plots...")

# res_s.plot_events(s='pos', bin_ranges= [[dia_min,dia_max]], event_list= bse_list_summer, commony=sharey, T_roll='24h')
# plt.savefig(os.path.join(results_dir, "all_pos-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.plot_events(s='neg', bin_ranges= [[dia_min,dia_max]], event_list= bse_list_summer, commony=sharey, T_roll='24h')
# plt.savefig(os.path.join(results_dir, "all_neg-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.scatter_values('pos', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_dtemp_pos.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.scatter_values('neg', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_dtemp_neg.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.scatter_values('pos', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_wind_pos.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.scatter_values('neg', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_wind_neg.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.scatter_values('pos', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_temperature_pos.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.scatter_values('neg', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll)
# plt.savefig(os.path.join(results_dir, "scatter_temperature_neg.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.scatter_WT('pos', bin_ranges=bin_all, commony=False, ras=True, pollution=True)
# plt.savefig(os.path.join(results_dir, "scatter_WT_pos.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# res_s.scatter_WT('neg', bin_ranges=bin_all, commony=False, ras=True, pollution=True)
# plt.savefig(os.path.join(results_dir, "scatter_WT_neg.png"), dpi=qual, bbox_inches='tight')
# plt.close()

# print(f"Global period plots are saved in {results_dir}")