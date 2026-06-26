import pandas as pd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ion_formation_rate3 import IonFormation as ifr

# ---- Study settings ---------------------------------------------------------
    # Time settings -------------------------
start_w = '2019-10-15 00:00:00'		# '2019-11-26 00:00:00'
end_w = '2020-10-01 00:00:00'		# '2019-12-09 00:00:00'
# start_w = '2020-06-18 00:00:00'
# end_w = '2020-06-25 00:00:00'

npf_datetime_list_text = [['2019-12-10 02:15:00', '2019-12-10 06:45:00'],
					['2019-12-02 14:00:00', '2019-12-06 04:00:00'],   # Qualitatively determined blowing snow events
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

npf_datetime_list_text1 = [['2019-12-02 00:00:00', '2019-12-06 00:00:00'],   #   Bergner et al. BSEs
						  ['2019-12-07 00:00:00', '2019-12-10 00:00:00'],
						  ['2020-01-15 00:00:00', '2020-01-16 00:00:00'],
						  ['2020-01-26 00:00:00', '2020-01-27 00:00:00'],
						  ['2020-02-02 00:00:00', '2020-02-04 00:00:00'],
						  ['2020-02-12 00:00:00', '2020-02-16 00:00:00'],
						  ['2020-02-18 00:00:00', '2020-02-22 00:00:00']]

# npf_datetime_list_text2 = [['2019-10-28 08:00:00', '2019-10-28 12:30:00'],	# Matt's events
# 						   ['2029-11-11 01:10:00', '2019-11-12 10:45:00'],
# 						   ['2019-11-15 19:15:00', '2019-11-16 01:00:00'],
# 						   ['2019-11-16 07:00:00', '2019-11-18 18:00:00'],	# polluted
# 						   ['2019-11-23 11:00:00', '2019-11-25 07:00:00'],
# 						   ['2019-12-02 15:30:00', '2019-12-06 01:00:00'],
# 						   ['2019-12-07 21:35:00', '2019-12-08 18:00:00'],	# polluted at the start of the event
# 						   ['2019-12-10 02:15:00', '2019-12-10 06:45:00'],
# 						   ['2020-01-01 08:00:00', '2020-01-03 01:00:00'],	# possibly polluted
# 						   ['2020-01-03 19:10:00', '2020-01-04 04:00:00'],	# SMPS labelled as pollution, but it is more likely blowing snow
# 						   ['2020-01-11 14:45:00', '2020-01-11 22:00:00'],	# a lot of pollution but with minimal influence?
# 						   ['2020-01-15 05:30:00', '2020-01-15 16:55:00'],	# Obvious short lived pollution spikes
# 						   ['2020-01-15 18:20:00', '2020-01-16 00:55:00'],	# Same as the previous one
# 						   ['2020-01-16 01:55:00', '2020-01-16 09:45:00'],	# Pollution flagging in the NAIS, but no evidence in the SMPS
# 						   ]

# Import events from Matthew's notes
pollution_remove = False			# if false, do not consider polluted events as events
df_events = pd.read_csv('Data/days-of-interest.csv', sep = ';')
df_events['start'] = pd.to_datetime(df_events['start'], format='ISO8601')
df_events['end'] = pd.to_datetime(df_events['end'], format='ISO8601')

if pollution_remove == True:
	df_events = df_events.loc[df_events['Pollution'] == False, :]

bse_list = df_events[['start', 'end']].values.tolist()
bse_datetime = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in bse_list] # For plot_events


    # Physics settings
temperature = 298          # [K], if None, met_data considered, else considered as constant (298K was default)
pressure = 101.3             # [kPa], if None, met_data considered, else considered as constant (101.3 was default)

dia_min = .75               # diameter window (from 0.75 to 31.62 [nm])
dia_max = 31.62             # (Using the 36.52 and 42.17 bins break the coag loss function (they are empty anyway). If the bins are wanted, uncommenting the NaN filter line in the function is required)

roll_period = None          # i.e '2h', if not None, apply a rolling median over the time given to smooth the data
diff_order = 2              # to compute dN/dt (see _diff function in the class)

    # Plot settings
def all_bin_size_by_four(bins):
	"""Make a list with all size bins suitable for plot functions"""
	bin_ranges = []
	for i in range(4, len(bins), 4):
		ranges  = [(bins[i-4], bins[i-4]), (bins[i-3], bins[i-3]), (bins[i-2], bins[i-2]), (bins[i-1], bins[i-1])]
		bin_ranges += [ranges]
	if len(bins)%4 != 0:
		nb_left_bins = len(bins)%4
		last_range = [(bins[-nb_left_bins], bins[-nb_left_bins]), (bins[-nb_left_bins+1], bins[-nb_left_bins+1]), (bins[-nb_left_bins+2], bins[-nb_left_bins+2])]
		bin_ranges += [last_range]
	return bin_ranges

def all_bin_size(bins):
	bin_ranges = []
	for size in bins:
		bin_ranges.append((size, size))
	return bin_ranges

bins = [0.75,  0.87,   1.0,  1.15,  1.33,  1.54,  1.78,  2.05,  2.37,  2.74,
		3.16,  3.65,  4.22,  4.87,  5.62,  6.49,   7.5,  8.66,  10.0, 11.55,
		13.34,  15.4, 17.78, 20.54, 23.71, 27.38, 31.62]

bin_ranges = [(0.75,  31.62), (2.05,  2.74), (3.16, 7.5), (8.66,  31.62)]   # for grouped subplots
bin_all = all_bin_size(bins)    # for unique bin subplots
bin_all = bin_all[0:19] # + [(11.55, 31.62)]

qual = 150              # Output plots quality
sharey = False          # Share y-axis when subplotting (not on heat map)
ylogscale = False       # log scale on y-axis
qual = 150              # Output plots quality

# ---------------------------------------------------------------------------
def banana_plot(psd_data, colorbar_max_lim=2000.0, ymin=3, ymax=550, cmap='viridis', title=None):
    
    psd = psd_data.copy()
    # need to add an additional time index so that the last row of real data is plotted
    freq = psd.index.to_series().diff().min()
    psd.loc[psd.index.max() + freq] = None
    
    #transpose the binned smps data for plotting
    transposed_data = psd.T
    
    #extract diameters from the psd dataframe (only works when using raw data loaded using fileloader.py)
    dp = psd.columns.values.astype(float)
    
    #generate plot
    fig, ax = plt.subplots()

    #image = ax.pcolormesh(psd.index, dp, transposed_data+1, norm=colors.LogNorm(), vmin=1, vmax=colorbar_max_lim, cmap=cmap )
    image = ax.pcolormesh(psd.index, dp, transposed_data+1, norm=colors.LogNorm(vmin=1, vmax=colorbar_max_lim), cmap=cmap )
    
    ax.set_title(title)
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Particle Diameter [nm]')
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_yscale('log')
    ax.grid(True, which='both', axis='both', linestyle='--', 
            color='k', linewidth=0.8)

    cbar = fig.colorbar(image,  pad = 0.1)
    cbar.set_label('dN/dlogDp [$cm^{-3}$]')
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
smps_10min_win = data_dic['smps'].loc[start_w:end_w]								# Not used !!!
nais_part_pos_10min_w = data_dic['nais_part_pos_file'].loc[start_w:end_w]
nais_ion_neg_10min_w = data_dic['nais_ion_neg_file'].loc[start_w:end_w]
nais_ion_pos_10min_w = data_dic['nais_ion_pos_file'].loc[start_w:end_w]
met_10min_w = data_dic['met'].loc[start_w:end_w]

# merge nais and smps
bin_min_smps = dia_max + .01	# to make sure not to have twice the same column
smps_10min_tomerge = smps_10min_win.loc[:, bin_min_smps:1000]
nais_smps_part = pd.concat([nais_part_pos_10min_w, smps_10min_tomerge], axis = 1)

# Cut of the weird values
def remove_spikes2(df, window='30min', threshold=5):
    """Replace values deviating more than threshold * local_std from the rolling median with NaN"""
    row_sum = df.sum(axis=1)
    rolling_med = row_sum.rolling(window=window, center=True, min_periods=1).median()
    rolling_std = row_sum.rolling(window=window, center=True, min_periods=1).std()
    outlier_mask = (row_sum - rolling_med).abs() > threshold * rolling_std
    outlier_mask_2d = pd.DataFrame(
        np.tile(outlier_mask.values[:, None], (1, df.shape[1])),
        index=df.index,
        columns=df.columns
    )
    return df.where(~outlier_mask_2d, other=np.nan)

def remove_spikes(df, threshold = 20000):
	mask = df.sum(axis = 1) > threshold
	df_clean = df.copy()
	df_clean.loc[mask] = np.nan
	return df_clean

nais_part_pos_10min_w = remove_spikes(nais_part_pos_10min_w, threshold = 1*10**6)
nais_ion_neg_10min_w  = remove_spikes(nais_ion_neg_10min_w)
nais_ion_pos_10min_w  = remove_spikes(nais_ion_pos_10min_w)

print("\t Data loaded, computing the results...")

res_w = ifr(nais_smps_part, nais_ion_pos_10min_w, nais_ion_neg_10min_w, met_10min_w, df_events= df_events,
			low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
			diff_order=diff_order, smooth_window=roll_period)
print("\t Results computed in the instance res_w")
# -------------------------------------------------------------------------------------------

# # ---- Slice datasets on one event period and compute results -----------------------------
event_number = 6
event_dates = bse_datetime[event_number]
start_ev = '2020-06-19 00:00:00'	#event_dates[0] 		# '2019/12/20 00:00:00'
end_ev = '2020-06-23 00:00:00'		#event_dates[1] 		# '2019/12/21 00:00:00'

nais_smps_part_ev = nais_smps_part.loc[start_ev:end_ev]
nais_part_pos_10min = data_dic['nais_part_pos_file'].loc[start_ev:end_ev]
nais_ion_neg_10min = data_dic['nais_ion_neg_file'].loc[start_ev:end_ev]
nais_ion_pos_10min = data_dic['nais_ion_pos_file'].loc[start_ev:end_ev]
met_10min = data_dic['met'].loc[start_ev:end_ev]

res = ifr(nais_smps_part_ev, nais_ion_pos_10min, nais_ion_neg_10min, met_10min, df_events=df_events,
			low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
			diff_order=diff_order, smooth_window=roll_period)
print("The instance containing the result has been created (res)")
# ---------------------------------------------------------------------------------------

# ---- Characterize all events -----------------------------------------------------------
# study_poll = True
# df_sel = df_events if study_poll else df_events.loc[df_events['Pollution'] == False]
# bse_list = df_sel.loc[:, ['start', 'end', 'Event Type', 'Pollution']].values.tolist()
# binmin = 1.
# binmax = 10.

# def time_average(Q):
# 	t_seconds = (Q.index - Q.index[0]).total_seconds().to_numpy()
# 	integral = np.trapezoid(Q.to_numpy(), t_seconds)   # [#/cm^3], total ions formed per cm3 over the event
# 	total_time = t_seconds[-1] - t_seconds[0]

# 	return integral / total_time

# res_dict = {}
# event_info = {}
# for start_ev, end_ev, event_type, poll in bse_list:

# 	event_name = f"{start_ev.date()}_to_{end_ev.date()}"

# 	nais_smps_part_ev = nais_smps_part.loc[start_ev:end_ev]
# 	nais_ion_pos_10min_ev = nais_ion_pos_10min_w.loc[start_ev:end_ev]
# 	nais_ion_neg_10min_ev = nais_ion_neg_10min_w.loc[start_ev:end_ev]
# 	met_10min_ev = met_10min_w.loc[start_ev:end_ev]

# 	res = ifr(nais_smps_part_ev, nais_ion_pos_10min_ev, nais_ion_neg_10min_ev, met_10min_ev, df_events=df_events,
# 			low_dia=dia_min, high_dia=dia_max, temperature=temperature, pressure=pressure,
# 			diff_order=diff_order, smooth_window=roll_period)
	
# 	res_dict[event_name] = res
# 	event_info[event_name] = {'Event Type': event_type, 'Pollution': poll}

# reduced_dict = {}
# result_df = pd.DataFrame(columns=['Q_pos_mean', 'Q_neg_mean', 'median_wind', 'median_temp', 'Event Type', 'Pollution', 'color'])
# for name, res in res_dict.items():
# 	Q_pos = res.Q_snow_pos.loc[:, binmin:binmax].sum(axis=1).dropna()
# 	Q_neg = res.Q_snow_neg.loc[:, binmin:binmax].sum(axis=1).dropna()
# 	wind_median = res.met_df['true_wind_velocity'].median()
# 	temp_median = res.met_df['air_temperature'].median()

# 	info = event_info[name]
# 	color = 'green' if info['Event Type']=='npf' else 'tomato' if info['Pollution'] else 'blue'
# 	newrow = pd.DataFrame([{'Q_pos_mean' 	: time_average(Q_pos),
# 						 'Q_neg_mean'		: time_average(Q_neg),
# 						 'median_wind'		: wind_median,
# 						 'median_temp'		: temp_median,
# 						 'Event Type'		: info['Event Type'],
# 						 'Pollution'		: info['Pollution'],
# 						 'color'			: color}], index = [name])
# 	result_df = pd.concat([result_df, newrow], ignore_index=True)

# x_data = 'median_wind'
# plt.figure()
# for color, label in [('green', 'NPF'), ('tomato', 'Polluted'), ('blue', 'Other')]:
#     subset = result_df[result_df['color'] == color]
#     plt.scatter(subset[x_data], subset['Q_pos_mean'], color=color, label=label)

# plt.xlabel(x_data)
# plt.ylabel('Q_pos_mean')
# plt.legend()


# x_data = 'median_wind'
# plt.figure()
# plt.scatter(result_df[x_data], result_df['Q_pos_mean'])
# plt.show()
# -------------------------------------------------------------------------------------------------



# res_w.boxplot_events('pos', x_data='wind', bin_ranges=[(1., 10.)], event_list=bse_list, width_frac=0.02, commony=False, showfliers=False)
# plt.show()














## To look at the noise for small bins. Remember to set an appropriate time window
# size = 0.75
# plt.figure()
# # plt.plot(res_w.pos_N_ion.loc[:, size], label = str(size))
# plt.plot(res_w.pos_N_ion.loc[:, 2.05:10.], label = )
# plt.legend()
# plt.gcf().autofmt_xdate()
# plt.title('Pos')

# plt.figure()
# plt.plot(res_w.neg_N_ion.loc[:, size], label = str(size))
# plt.plot(res_w.neg_N_ion.loc[:, 2.05], label = '2.05')
# plt.legend()
# plt.gcf().autofmt_xdate()
# plt.title('Neg')
# plt.show()
sizes = [.75, .87, 1., 1.54, 2.05, 10.]
start_noise = '2020-06-19 00:00:00'
end_noise = '2020-06-23 00:00:00'
resample_time = '2h'
df_pos = res_w.pos_N_ion.loc[start_noise:end_noise, sizes].rolling(window = resample_time, center = True).mean()
df_neg = res_w.neg_N_ion.loc[start_noise:end_noise, sizes].rolling(window = resample_time, center = True).mean()

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,6), sharex=True)
df_pos.plot(ax = ax1, alpha = 0.8)
ax1.set_title("pos")
df_neg.plot(ax = ax2, alpha = 0.8)
ax2.set_title("neg")

for ax in (ax1, ax2):
	ax.legend()
ax1.set_ylabel("Concentration ($cm^{-3}$)")
ax1.set_xlabel("Datetime")
fig.autofmt_xdate()
fig.suptitle("Concentrations of ions before, during and after midsummer event")

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,6))
res.plot_hm_conc(s= 'pos', bin_range=(.75, 31.62), ax=ax1)
res.plot_hm_conc(s= 'neg', bin_range=(.75, 31.62), ax=ax2)
# res_w.plot_hm_conc(s = 'pos', bin_range=(1.54, 31.62), vmaxi=500)
# res_w.plot_hm_conc(s = 'pos', bin_range=(.75, 31.62))

plt.show()