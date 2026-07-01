# exe_all compute the results for each npf events indicated in the npf_datetime_list_text
# It stores all figures and csvs in dedicated folder, in results_all/
# WARNING: it deletes all data in Results_all at each run !!!

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ion_formation_rate3 import IonFormation as ifr
from typing import Literal
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

    # Physics settings -----------------------
temperature = None          # [K], if None, met_data considered, else T considered as constant (298K was default)
pressure = None             # [kPa], if None, met_data considered, else P considered as constant (101.3 was default)

dia_min = 1.54               # diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62             # (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

roll_period = None          # '2h', if not None, apply a rolling median over the time given to smooth the data
diff_order = 2              # to compute dN/dt (see _diff function in the class)

    # Plot settings -----------------------------
study_poll = False			# Compute and plot 'per event' results for polluted events as well
add_ras = False 			# add non event values on scatter plots
add_poll = False			# add polluted event on scatter plots
add_npf = True              # add npf event in plots

sharey = False       # Share y-axis when subplotting (not on heat map)
ylogscale = False   # log scale on y-axis
qual = 150          # Output plots quality

save_csv = True

bins_all = [0.75,  0.87,   1.0,  1.15,  1.33,  1.54,  1.78,  2.05,  2.37,  2.74,
        3.16,  3.65,  4.22,  4.87,  5.62,  6.49,   7.5,  8.66,  10.0, 11.55,
        13.34,  15.4, 17.78, 20.54, 23.71, 27.38, 31.62]

def set_bin_all(all_bins, stop_bin = dia_max, dia_min = dia_min, group_big:bool = True, max_bins = 19):
    """Create a list with desired bins for plots.
    Limite the number of bins to max_bins for clarity"""
    if dia_min in all_bins:
        lower_bin = bins_all.index(dia_min)
    else:
        print('dia_min is not in the colums bins')
        lower_bin = 0

    if stop_bin in all_bins: 
        stop_bin_idx = all_bins.index(stop_bin)
        if stop_bin == all_bins[-1]:
            group_big = False
    elif group_big: stop_bin_idx = max_bins
    else: stop_bin_idx = len(all_bins)-1

    bins = all_bins[lower_bin:stop_bin_idx+1]
    bin_all = [(size, size) for size in bins]

    if group_big:
        bin_all += [(all_bins[stop_bin_idx+1], all_bins[len(all_bins)-1])]    # Group the bigger ones which give the same results for clearer plot

    return bin_all

bin_all = set_bin_all(bins_all, stop_bin=11.55, dia_min=dia_min, group_big=True)

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

# def give_resdict(start_w, end_w, start_s, end_s, data_dic, df_events = df_events, study_poll = study_poll):
# 	start = start_w
# 	end = end_s if end_s is not None else end_w
    
# 	# ---- Load data ------------------
# 	smps_10min_win = data_dic['smps'].loc[start:end]
# 	nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start:end]
# 	nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start:end]
# 	nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start:end]
# 	met_10min = data_dic['met'].loc[start:end]
    
# 	bin_min_smps = dia_max + .01	# to make sure not to have twice the same column
# 	smps_10min_tomerge = smps_10min_win.loc[:, bin_min_smps:1000]
# 	nais_smps_part = pd.concat([nais_part_pos_10min, smps_10min_tomerge], axis = 1)
# 	if spikes_remove == True:
# 		nais_part_pos_10min = remove_spikes(nais_part_pos_10min, threshold = 1*10**6)
# 		nais_ion_neg_10min  = remove_spikes(nais_ion_neg_10min)
# 		nais_ion_pos_10min  = remove_spikes(nais_ion_pos_10min)
    
# 	res_w = ifr(nais_smps_part, nais_ion_pos_10min, nais_ion_neg_10min, met_10min, df_events=df_events,
#                 low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
#                 diff_order=diff_order, smooth_window=roll_period)
def build_results_df(data_dic, level_names=('member', 'bin')):
    """Turn a dic_pos/dic_neg style dict into a single MultiIndex DataFrame.
    Rows: time index. Columns: (equation member, size bin)."""
    df = pd.concat(data_dic, axis=1)
    df.columns = df.columns.set_names(level_names)
    return df

def all_res(start, end, result_dir, data_dic = data_dict, df_events = df_events, study_poll = study_poll):
    """Compute and save plots all results according to the settings (work with global variables)"""
    print(f"\n Analysis from {start} to {end}")
    # ---- clean result folder --------------------
    # if os.path.exists(result_dir):
    #     shutil.rmtree(result_dir)
    # os.makedirs(result_dir)

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
    # res_dict['global'] = res_w
    # print("\t Results computed. Saving the plots...")

    # res_w.plot_events(s='pos', bin_ranges= [[dia_min,dia_max]], study_poll=True, commony=sharey, T_roll='24h')
    # plt.savefig(os.path.join(result_dir, "all_pos-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.plot_events(s='neg', bin_ranges= [[dia_min,dia_max]], study_poll=True, commony=sharey, T_roll='24h')
    # plt.savefig(os.path.join(result_dir, "all_neg-ion-conc_events.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.scatter_values('pos', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    # plt.savefig(os.path.join(result_dir, "scatter_dtemp_pos.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.scatter_values('neg', x_data='dtemp', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    # plt.savefig(os.path.join(result_dir, "scatter_dtemp_neg.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.scatter_values('pos', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    # plt.savefig(os.path.join(result_dir, "scatter_wind_pos.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.scatter_values('neg', x_data='wind', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    # plt.savefig(os.path.join(result_dir, "scatter_wind_neg.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.scatter_values('pos', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    # plt.savefig(os.path.join(result_dir, "scatter_temperature_pos.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.scatter_values('neg', x_data='temperature', bin_ranges=bin_all, ras=add_ras, pollution=add_poll, npf=add_npf)
    # plt.savefig(os.path.join(result_dir, "scatter_temperature_neg.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.scatter_WT('pos', bin_ranges=bin_all, commony=False, ras=add_ras, pollution=add_poll, npf=add_npf)
    # plt.savefig(os.path.join(result_dir, "scatter_WT_pos.png"), dpi=qual, bbox_inches='tight')
    # plt.close()

    # res_w.scatter_WT('neg', bin_ranges=bin_all, commony=False, ras=add_ras, pollution=add_poll, npf=add_npf)
    # plt.savefig(os.path.join(result_dir, "scatter_WT_neg.png"), dpi=qual, bbox_inches='tight')
    # plt.close()
    
    if save_csv: 
        df_pos = build_results_df(res_w.dic_pos)
        df_neg = build_results_df(res_w.dic_neg)
        
        df_pos.to_csv(os.path.join(result_dir, "global_pos.csv"))
        df_neg.to_csv(os.path.join(result_dir, "global_neg.csv"))

    # print(f"\t Global period plots are saved in {result_dir}")
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

    #     # ---- make the directory to the dedicated folder
    #     event_dir = os.path.join(result_dir, event_name)
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

    #     res.volume_plot('pos', bin_ranges=bin_all, commony=False)
    #     plt.savefig(os.path.join(event_dir, "volume_production_pos.png"), dpi=qual, bbox_inches='tight')
    #     plt.close()

    #     res.volume_plot('neg', bin_ranges=bin_all, commony=False)
    #     plt.savefig(os.path.join(event_dir, "volume_production_neg.png"), dpi=qual, bbox_inches='tight')
    #     plt.close()

    #     note = df_events.loc[df_events['start'] == start_ev, 'notes'].values[0]
    #     with open(os.path.join(event_dir, "notes.txt"), 'w') as f:
    #         f.write(str(note))
    #     print(f"{event_dir} done")
    # print(f"All event results are saved in {result_dir} in their dedicated folder")

    return res_dict
# ----------------------------------------------------------------------

# ---- Compute winter results ----------------------------
df_events_w = df_events.copy()
df_events_w = df_events_w[(df_events_w['start'] > pd.to_datetime(start_w)) & (df_events_w['end'] < pd.to_datetime(end_w))]

res_dict_w = all_res(start=start_w, end = end_w, result_dir="Results_winter", data_dic=data_dict, df_events= df_events_w)


# ---- Compute summer results -----------------------
df_events_s = df_events.copy()
df_events_s = df_events_s[(df_events_s['start'] > pd.to_datetime(start_s)) & (df_events_s['end'] < pd.to_datetime(end_s))]

res_dict_s = all_res(start=start_s, end = end_s, result_dir= "Results_summer", data_dic=data_dict, df_events= df_events_s)




# ---- Compute general results ------------------------
bin_min, bin_max = (dia_min, dia_max)
def time_average(Q):
    t_seconds = (Q.index - Q.index[0]).total_seconds().to_numpy()
    integral = np.trapezoid(Q.to_numpy(), t_seconds)   # [#/cm^3], total ions formed per cm3 over the event
    total_time = t_seconds[-1] - t_seconds[0]

    return integral / total_time

def plot_global(res_dicts:tuple, radiation:tuple, xval:Literal['wind', 'temp'] = 'wind', bin_range=(dia_min, dia_max)):
    if xval == 'wind':
        key = 'true_wind_velocity'
        xlabel = 'Wind speed ($m\\cdot s^{-1}$)'
    elif xval == 'temp':
        key = 'air_temperature'
        xlabel = 'Temperature (°C)'
    else: raise ValueError("xval must be 'wind' or 'temp'")
    bin_min, bin_max = bin_range
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,6), sharex=True)
    for rad, res_dict in zip(radiation, res_dicts):
        marker = 'x' if rad else 'o'
        for name, event_res in res_dict.items():
            Q_pos = event_res.Q_snow_pos.loc[:, bin_min:bin_max].sum(axis=1).dropna()
            Q_pos_int = time_average(Q_pos)
            Q_neg = event_res.Q_snow_neg.loc[:, bin_min:bin_max].sum(axis=1).dropna()
            Q_neg_int = time_average(Q_neg)
            
            x_data = event_res.met_df[key].median()
            theresnpf = event_res.theresnpf
            color = event_res.coldict['npf'] if theresnpf else event_res.coldict['events']
            label = 'NPF event' if theresnpf else 'BSE'
            ax2.scatter(x_data, Q_pos_int, color = color, marker = marker, label = label)
            ax1.scatter(x_data, Q_neg_int, color = color, marker = marker, label = label)
            
    legend_elements = [
        ax1.scatter([], [], marker='x', color='k', label='Summer (res_dict_s)'),
        ax1.scatter([], [], marker='o', color='k', label='Winter (res_dict_w)'),
        ax1.scatter([], [], marker='s', color=res_dict_w[next(iter(res_dict_w))].coldict['npf'], label='NPF event'),
        ax1.scatter([], [], marker='s', color=res_dict_w[next(iter(res_dict_w))].coldict['events'], label='BSE'),
    ]
    ax2.legend(handles=legend_elements)
    ax1.set_title("Negative Ions")
    ax2.set_title("Positive Ions")
    ax1.set_ylabel("Mean Production rate ($cm^{-1}\\,s^{-1}$)")
    ax1.set_xlabel(xlabel)
    ax2.set_xlabel(xlabel)
    fig.suptitle("'Mean' production rate per event")
    
plot_global(res_dicts=(res_dict_s, res_dict_w), radiation=(True, False), xval='wind')
plot_global(res_dicts=(res_dict_s, res_dict_w), radiation=(True, False), xval='temp')
plt.show()

