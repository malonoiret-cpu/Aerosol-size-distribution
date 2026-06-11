# this class is based on ion_formation_rate2.py, and aim to give results according to the size bins

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

class IonFormation:
    def __init__(self, particle_psd: pd.DataFrame, pos_ion_psd: pd.DataFrame, neg_ion_psd: pd.DataFrame, met_df: pd.DataFrame, df_events : pd.DataFrame = None,       \
			  		low_dia=None, high_dia=None, pressure = None, temperature = None, alpha = 1.6e-6, chi = 0.01e-6, rho = 0.00183,             \
                    diff_order: int = 2, smooth_window = None):
        # define constants
        self.BOLTZMANN = 1.380658e-23 	# [m**2 kg/s**2 K]
        self.SUTHERLAND = 110.4			# sutherland correction [K]
        self.MFP_REF = 6.730e-8			# mean free path [m]
        self.T_REF = 296.15				# Temperature [K]
        self.P_REF = 101.3				# reference pressure [kPa]
        self.VISCOSITY_REF = 1.83245e-5	# reference dynamic viscocity [kg/m*s]
        self.rho = rho					# [kg/cm**3], density of the particles, assumed to be sulfuric acid for default value

        # Ion coefficients, default values taken from Kulmala et al., 2012
        self.alpha = alpha				# ion-ion recombination coefficient [cm**3/s]
        self.chi = chi					# ion-aerosol attachment coefficient [cm**3/s]

        ## Set constant temperature and pressure. If None, met data are considered
        self.pressure = pressure            # 101.3
        self.pressure_series = met_df['air_pressure'] * 0.1             # [kPa]
        self.temperature = temperature      # 298.
        self.temperature_series = met_df['air_temperature'] + 273.15    # [K]
        
        # Variables
        ## ---- Mesurement data ------------------------------------
        if not isinstance(pos_ion_psd, pd.DataFrame):			# to stop the computation if ion_psd is not a data frame
            raise TypeError("df must be a pandas DataFrame")
        self.pos_ion_psd = pos_ion_psd
        if not isinstance(neg_ion_psd, pd.DataFrame):			# to stop the computation if ion_psd is not a data frame
            raise TypeError("df must be a pandas DataFrame")
        self.neg_ion_psd = neg_ion_psd.reindex(pos_ion_psd.index)   # Re-index particle psd with pos ion psd to make sure they have the same time index for calculations (they are resampled anyway)
        if not isinstance(particle_psd, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.particle_psd = particle_psd.reindex(pos_ion_psd.index) # Re-index particle psd with pos ion psd to make sure they have the same time index for calculations

        self.met_df = met_df
        ## ----------------------------------------------------------

        ## Diameter range considered
        if low_dia==None:
            self.low_dia = self.pos_ion_psd.columns[0]
        else: self.low_dia = low_dia
        if high_dia==None:
            self.high_dia = self.pos_ion_psd.columns[-1]
        else: self.high_dia = high_dia

        ## ---- Slice the psd to the range considered---------------- should consider the [:,-1] ?
            # pos and neg ion psd in the range of interest
        self.nucmode_pos_ion_psd = self.pos_ion_psd.loc[:, self.low_dia:self.high_dia]  # slice the size distribution down to the selected size bin of interest
        self.pos_N_ion = self.nucmode_pos_ion_psd                                       # [#/cm**3], the concentration of pos ions between dp_min and dp_max

        self.nucmode_neg_ion_psd = self.neg_ion_psd.loc[:, self.low_dia:self.high_dia]
        self.neg_N_ion = self.nucmode_neg_ion_psd # [#/cm**3], matrix (T,N)

            # pos and neg ion smaller than the range of interest
        self.pos_ion_smaller_psd = self.N_smaller(self.pos_ion_psd).loc[:, self.low_dia:self.high_dia]
        self.N_pos_ion_smaller = self.pos_ion_smaller_psd # concentration [#/cm**3], matrix (T,N)

        self.neg_ion_smaller_psd = self.N_smaller(self.neg_ion_psd).loc[:, self.low_dia:self.high_dia]
        self.N_neg_ion_smaller = self.neg_ion_smaller_psd # concentration [#/cm**3], matrix (T,N)

            # particle psd in the range of interest
        self.nucmode_particle_psd = self.particle_psd.loc[:, self.low_dia:self.high_dia] # slice the size distribution down to the size range of interest
        self.N_particle = self.nucmode_particle_psd # [#/cm**3], the concentration of particles between dp_min and dp_max, for each been
        ## -----------------------------------------------------------

        ## ---- Now the terms of the equation for Q_snow can be calculated ---------------------------------------------------------------
            ## Compute the members of the Q_snow_pos equation
        self.dNdp_dt_pos_ion = self._diff(self.pos_N_ion, order=diff_order)
        self.pos_coag_loss_term = self.calc_coag_loss(ion_psd = self.pos_ion_psd)[:-1] * self.pos_N_ion[:-1]    # T, P dependent
        self.pos_growth_rate_term = 0
        self.pos_alpha_term = self.alpha * self.pos_N_ion[:-1] * self.N_neg_ion_smaller[:-1]
        self.pos_chi_term = self.chi * self.N_particle[:-1] * self.N_pos_ion_smaller[:-1]

            ## Compute the members of the Q_snow_neg equation
        self.dNdp_dt_neg_ion = self._diff(self.neg_N_ion, order=diff_order)
        self.neg_coag_loss_term = self.calc_coag_loss(ion_psd = self.neg_ion_psd)[:-1] * self.neg_N_ion[:-1]
        self.neg_growth_rate_term = 0
        self.neg_alpha_term = self.alpha * self.neg_N_ion[:-1] * self.N_pos_ion_smaller[:-1]
        self.neg_chi_term = self.chi * self.N_particle[:-1] * self.N_neg_ion_smaller[:-1]
        
        self.Q_snow_pos = self.Q_snow_calc(s = 'pos')
        self.Q_snow_neg = self.Q_snow_calc(s = 'neg')
        ## -------------------------------------------------------------------------------------------------------------------------------

        # ---- Smooth if asked ------------------------------------------------------------------
        self.smooth_window = smooth_window  # Used in plot events
        if smooth_window != None:
            print(f"The results have been smoothed, taking the median over a {smooth_window} window.")
            self.Q_snow_pos = self.Q_snow_pos.rolling(window = smooth_window, center = True).median()
            self.dNdp_dt_pos_ion = self.dNdp_dt_pos_ion.rolling(window = smooth_window, center = True).median()
            self.pos_coag_loss_term = self.pos_coag_loss_term.rolling(window = smooth_window, center = True).median()
            self.pos_growth_rate_term = 0
            self.pos_alpha_term = self.pos_alpha_term.rolling(window = smooth_window, center = True).median()
            self.pos_chi_term = self.pos_chi_term.rolling(window = smooth_window, center = True).median()

            self.Q_snow_neg = self.Q_snow_neg.rolling(window = smooth_window, center = True).median()
            self.dNdp_dt_neg_ion = self.dNdp_dt_neg_ion.rolling(window = smooth_window, center = True).median()
            self.neg_coag_loss_term = self.neg_coag_loss_term.rolling(window = smooth_window, center = True).median()
            self.neg_growth_rate_term = 0
            self.neg_alpha_term = self.neg_alpha_term.rolling(window = smooth_window, center = True).median()
            self.neg_chi_term = self.neg_chi_term.rolling(window = smooth_window, center = True).median()

        # store the results in dic for plots
        self.dic_pos = {
                r"$Q_{\mathrm{snow}}$": self.Q_snow_pos,
                r"$\partial N / \partial t$": self.dNdp_dt_pos_ion,
                r"Coagulation loss": self.pos_coag_loss_term,
                r"$\alpha$ term": self.pos_alpha_term,
                r"$\chi$ term": self.pos_chi_term,
            }
        self.dic_neg = {
                r"$Q_{\mathrm{snow}}$": self.Q_snow_neg,
                r"$\partial N / \partial t$": self.dNdp_dt_neg_ion,
                r"Coagulation loss": self.neg_coag_loss_term,
                r"$\alpha$ term": self.neg_alpha_term,
                r"$\chi$ term": self.neg_chi_term,
            }
        
        threshold = 1.0  # cm-3, adjust to what makes physical sense
        self.dic_ratio = {}
        for name in self.dic_pos:
            pos = self.dic_pos[name]
            neg = self.dic_neg[name]
            ratio = neg / pos
            ratio = ratio.where(pos.abs() >= threshold, other=np.nan)  # mask near-zero denominators
            self.dic_ratio[name] = ratio
        
        # ---- event tags --------------
        self.df_events = df_events
        self.event_tags = self.tag_events()


    def N_smaller(self, psd):
        """Compute the number of smaller particle than a bin for each bin size (cumsum)
        Input: pos or neg ion psd"""
        full_cumsum = psd.cumsum(axis = 1)
        full_cumsum_shifted = full_cumsum.shift(1, axis=1).fillna(0)    # The first bin is filled with 0s, the second with the concentration of the first bin, the third the sum of the two first...
        return full_cumsum_shifted
    
    def _diff(self, df: pd.DataFrame, order: int = 2, fb: Literal['backward', 'forward'] = 'backward'):
        """
        Compute dN/dt using different methods.
        order=2 : standard 2-point 'fb' difference (default value when calling the class)
            -> fb determine if the difference is backaward (N(t)-N(t-1)) or forward (N(t+1) - N(t))
        order=3 : 3-point central difference
        order=5 : 5-point central difference
        Returns a DataFrame aligned on the interior time index.
        """
        N = df.values
        t = df.index.astype(np.int64).to_numpy() / 1e9  # timestamps in seconds

        if order == 2:
            dN = N[1:] - N[:-1]
            dt = np.diff(t)[:, None]
            idx = df.index[1:] if fb == 'backward' else df.index[:-1]

        elif order == 3:
            dN = N[2:] - N[:-2]
            dt = (t[2:] - t[:-2])[:, None]
            idx = df.index[1:-1]

        elif order == 5:
            dN = (-N[4:] + 8*N[3:-1] - 8*N[1:-3] + N[:-4])
            dt = (12 * (t[2:-2] - t[1:-3]))[:, None]  # 12 * dt (uniform spacing assumed)
            idx = df.index[2:-2]

        else:
            raise ValueError("order must be 2, 3, or 5")

        return pd.DataFrame(dN / dt, index=idx, columns=df.columns)

    def tag_events(self):
        """Gives a tag for each time stamps (based on Q_snow_pos index) as follow:
            - NaN: no event
            - event: during an event without pollution
            - event_poll: during an event qualified as polluted
        If no events are given, no labels are applied."""
        tags = pd.Series(np.nan, index=self.Q_snow_pos.index, dtype=object)

        if self.df_events is not None:
            for _, row in self.df_events.iterrows():
                mask = (tags.index >= row['start']) & (tags.index <= row['end'])
                label = 'event_poll' if row['Pollution'] else 'event'
                tags[mask] = label
        
        return tags

    def mean_free_path_calc(self, T, P):
        """Compute the mean free path from the reference (at T = 296.15 [K], P = 101.3 [kPa]; MFP_REF = 6.730e-8 [m])
        Output: float: mean free path [m]"""
        mfp = self.MFP_REF * (self.P_REF / P) * (T / self.T_REF)                \
            * ((1.0 + self.SUTHERLAND / self.T_REF)/(1.0 + self.SUTHERLAND / T)) # [m])
        return mfp
	
    def viscosity_calc(self, T):
        """Compute the viscosity (simplification of the TSI equation, given by Ernie Lewis)
        Input: temperature [K] (default 298 K)
        Output: float: viscosity"""
        return self.VISCOSITY_REF * (T / self.T_REF)**0.78

    def calc_coag_loss(self, ion_psd):
        """Compute the coagulation loss. First, all units are converted to centimeters to compute CoagS in cm. Then, the total coag loss for each time step is calculated.
        Inputs:	- the pos or neg ion psd
        Outputs:	- array containing all total CoagL for each time step
                        -> result differs to coag_loss_1 with at the 10^-11th decimal"""

        # convert all dimensions for the coagulation losses to cm
        boltzmann = self.BOLTZMANN * 10000 		# units for boltzmann constant is in [m**2 kg/s**2 K], multiplied by 10000 to convert to [cm**2*kg/s**2*K]
        

        # create a subset of the PSD over which the CoagS is calculated, based on input diameters
        nuc_mode_psd = ion_psd.loc[:, self.low_dia:self.high_dia]
        particle_psd = self.particle_psd.loc[:, self.low_dia:self.high_dia]

        coag_loss_all_sum = []	# list to save the cumulative coagulation loss in the nuc mode for each scan (final list)

        # Create arrays containing all bins for both nuc (i) and part (j)
        bins_nuc = nuc_mode_psd.columns.to_numpy()*1e-7			# array, shape (N-i, ), convert nm to cm for calculations
        bins_part = particle_psd.columns.to_numpy()*1e-7	# array, shape (M-i, ), convert nm to cm for calculations

        N = len(bins_nuc)
        M = len(bins_part)

        # Reshape to get matrices (N,1) and (1,M), so product gives (N,M)
        d1 = bins_nuc[:, None]   # shape (N, 1)
        d2 = bins_part[None, :]  # shape (1, M)
        
        # upper triangle mask: keeps only j >= i pairs, zeros the rest (replaces j loop range(i, M))
        triu_mask = np.triu(np.ones((N, M), dtype=bool))  # shape (N, M)

        # self-coagulation mask: diagonal where i == j
        diag_idx = np.arange(min(N, M)) # diagonal indices to apply the mask later

        # loop through each SMPS scan (every timestamp) associated with the NPF event
        coag_loss_all_sum = []	# list to save the cumulative coagulation loss in the nuc mode for each scan (final list)

        for t in range(len(ion_psd.index)):
            
            timestamp = ion_psd.index[t]
            T = self.temperature if self.temperature is not None else self.temperature_series.loc[timestamp]
            P = self.pressure if self.pressure is not None else self.pressure_series[timestamp]

            mfp = self.mean_free_path_calc(T=T, P=P) * 100 	# mean free path multiplied by 100 to get it in [cm]
            mu = self.viscosity_calc(T=T) / 100 		# viscosity, divide by 100 to convert [kg/m*s] to [kg/cm*s]

            ## Compute values for Kij
            # slip correction factor
            Cc1 = 1.0 + (2.0 * mfp / d1) * (1.257 + 0.4 * np.exp(-1.1 * d1 / (2.0 * mfp)))  # (N, 1)
            Cc2 = 1.0 + (2.0 * mfp / d2) * (1.257 + 0.4 * np.exp(-1.1 * d2 / (2.0 * mfp)))  # (1, M)

            # diffusivity [cm**2/s]
            D1 = (boltzmann * T * Cc1) / (3.0 * np.pi * mu * d1)  # (N, 1)
            D2 = (boltzmann * T * Cc2) / (3.0 * np.pi * mu * d2)  # (1, M)

            # mass of particles [kg]
            m1 = (1.0 / 6.0) * np.pi * (d1**3.0) * self.rho		# (N, 1)
            m2 = (1.0 / 6.0) * np.pi * (d2**3.0) * self.rho		# (1, M)

            # average velocity [cm/s]
            c_bar1 = ((8.0 * boltzmann * T) / (np.pi * m1))**0.5  # (N, 1)
            c_bar2 = ((8.0 * boltzmann * T) / (np.pi * m2))**0.5  # (1, M)

            # mean free path [cm]
            l1 = 8.0 * D1 / (np.pi * c_bar1)  # (N, 1)
            l2 = 8.0 * D2 / (np.pi * c_bar2)  # (1, M)

            # correction terms
            g1 = (1.0 / (3.0 * d1 * l1)) * ((d1 + l1)**3.0 - (d1**2.0 + l1**2.0)**(3.0/2.0)) - d1  # (N, 1)
            g2 = (1.0 / (3.0 * d2 * l2)) * ((d2 + l2)**3.0 - (d2**2.0 + l2**2.0)**(3.0/2.0)) - d2  # (1, M)

            # coagulation coefficient [cm**3/s] matrix with shape (N, M) (rate constante for collisions)
            Kij = 2.0 * np.pi * (D1 + D2) * (d1 + d2) / 										\
                ((((d1 + d2) / (d1 + d2 + 2.0 * (g1**2.0 + g2**2.0)**0.5)) 						\
                + (8.0 * (D1 + D2) / ((c_bar1**2.0 + c_bar2**2.0)**0.5 * (d1 + d2))))**-1.0)

            conc_nuc  = nuc_mode_psd.iloc[t].to_numpy()          # shape (N,)
            conc_part = particle_psd.iloc[t].to_numpy()      # shape (M,)

            # collision rate: Kij * conc_i * conc_j : shape (N, M)
            Jij = Kij * conc_nuc[:, None] * conc_part[None, :]

            # apply upper triangle mask (j >= i only)
            Jij = np.where(triu_mask, Jij, 0.0)

            # self-coagulation correction on the diagonal (i == j)
            Jij[diag_idx, diag_idx] /= 2.0

            # Jij = np.nan_to_num(Jij, nan=0.0)                       # fix the NaN issue (ignore NaNs in the sum)
            coag_loss_all_sum.append(Jij.sum(axis=1))

        return pd.DataFrame(
                np.array(coag_loss_all_sum),
                index=ion_psd.index,
                columns=nuc_mode_psd.columns
            )
    
    def Q_snow_calc(self, s: Literal['all', 'pos', 'neg'] = "all"):
        """Compute Q_snow"""
        Q_snow_pos = self.dNdp_dt_pos_ion       \
                    + self.pos_coag_loss_term   \
                    + self.pos_growth_rate_term \
                    + self.pos_alpha_term       \
                    - self.pos_chi_term
        
        Q_snow_neg = self.dNdp_dt_neg_ion       \
                    + self.neg_coag_loss_term   \
                    + self.neg_growth_rate_term \
                    + self.neg_alpha_term       \
                    - self.neg_chi_term
        
        if s == 'all':
            return Q_snow_pos, Q_snow_neg
        if s == 'pos':
            return Q_snow_pos
        if s == 'neg':
            return Q_snow_neg
        
    def calc_growth_rate(self):
        """Calculate the growth rate"""
        return 0
    
    def plot_events(self, s: Literal['pos', 'neg'], bin_ranges : list, event_list : list, commony = False, T_roll = None):
        """Plot concentration for each bin range given and the wind over time.
        Highlight the events studied with the given event list"""

        if s == 'pos':
            df_conc = self.pos_N_ion
            main_title = f"Positive ion concentration over the winter"
        elif s == 'neg':
            main_title = f"Negative ion concentration over the winter"
            df_conc = self.neg_N_ion
        else:
            raise ValueError("s must be 'pos' or 'neg'")
        
        df_wind = self.met_df['true_wind_velocity']
        df_rad = self.met_df['global_radiation']
        
        if self.smooth_window is not None:
            df_conc = df_conc.rolling(window=self.smooth_window, center = True).mean()
            df_wind = df_wind.rolling(window=self.smooth_window, center = True).mean()
            df_rad = df_rad.rolling(window=self.smooth_window, center = True).mean()
            if T_roll is not None:
                print(f"Warning: the rolling mean is applied with the window given to the class ({self.smooth_window}), not the window given when plotting ({T_roll})")
        elif T_roll is not None:
            df_conc = df_conc.rolling(window=T_roll, center = True).mean()
            df_wind = df_wind.rolling(window=T_roll, center = True).mean()
            df_rad = df_rad.rolling(window=T_roll, center = True).mean()

        nplots = len(bin_ranges)                #
        ncol = int(np.ceil(np.sqrt(nplots)))    # Design the subplot matrix
        nrow = int(nplots / ncol)               #
        fig, axs = plt.subplots(nrow,ncol, figsize = (ncol*12,nrow*5), sharex=True, sharey=commony, squeeze=False)

        for idx, (ax1, (lo, hi)) in enumerate(zip(axs.flatten(), bin_ranges)):
            
            col = idx%ncol

            ax2 = ax1.twinx()   # Plot the wind
            ax2.plot(df_wind, '-', alpha = 0.2, color = "#da6dd0", label = 'Wind velocity')
            if col == ncol -1:
                ax2.set_ylabel("Wind velocity ($m.s^{-1}$)", color = "#da6dd0")
                ax2.tick_params(axis='y', colors="#da6dd0")
            else : ax2.tick_params(axis='y', colors="#da6dd0")

            ax3 = ax1.twinx()   # Plot global radiation
            ax3.spines["right"].set_position(("axes", 1.1))  # offset so it doesn't overlap ax2
            ax3.plot(df_rad, alpha = 0.5, color = 'grey', label = "Global radiation ($W.m^{-2}$)")
            ax3.set_ylim(-500, 420)
            if col == ncol -1:
                ax3.set_ylabel("Global radiation ($W.m^{-2}$)", color = 'grey')
                ax3.tick_params(axis='y', colors='grey')
            else : ax3.tick_params(axis='y', colors='grey')

            ax1.plot(df_conc.loc[:, lo:hi].sum(axis=1), '-', color = 'blue', label = 'Concentration')
            ax1.set_ylabel("Concentration (dN/dlogDp)", color = 'blue')
            if col == 0:
                ax1.set_xlabel("DateTime")
            ax1.grid()
            subtitle = f"{lo} nm" if lo == hi else f"{lo} to {hi} nm"
            ax1.set_title(subtitle)

            for (start, end), ev_nb in zip(event_list, range(len(event_list))):     # Plot the wind events
                ax2.axvspan(xmin = start, xmax = end, color = "#087edf", alpha = 0.3)
                # ax2.text(start, np.max(df_wind), ev_nb)
                mid = start + (end - start) / 2                      # center of the span
                ypos = np.max(df_wind) * (1 if ev_nb % 3 == 0 else 0.95 if ev_nb%3 == 1 else 0.9)  # alternate height
                ax2.text(mid, ypos, str(ev_nb), ha='center', va='top')

        fig.suptitle(main_title)
        fig.autofmt_xdate()
        plt.tight_layout()

        
    def plot_hm_conc(self, s:Literal['pos','neg', 'ratio']='pos', vmini = None, vmaxi = None, cmap = "RdBu_r"):
        """Plot the heatmap of the concentrations over the time and the particle size"""
        if s == "pos":
            main_title = f"Positively charged particles ({self.low_dia} to {self.high_dia} nm)"
            df = self.pos_N_ion
        elif s == "neg":
            main_title = f"Negatively charged particles ({self.low_dia} to {self.high_dia} nm)"
            df = self.neg_N_ion
        elif s == "ratio":
            main_title = f"Negative / Positive ({self.low_dia} to {self.high_dia} nm)"
            df = self.neg_N_ion / self.pos_N_ion
            df = df.where(self.pos_N_ion.abs() >= 1, other=np.nan)  # mask near-zero denominators
        else:
            raise ValueError("s must be 'pos', 'neg' or 'ratio")
        
        wind_df = self.met_df['true_wind_velocity']

        if self.smooth_window is not None:  # smooth data if asked
            df = df.rolling(window=self.smooth_window, center = True).mean()
            wind_df = wind_df.rolling(window=self.smooth_window, center = True).mean()
        
        fig, ax1 = plt.subplots(figsize = (8,5))

        ax2 = ax1.twinx()
        ax2.spines["right"].set_position(("axes", 1.15))
        ax2.plot(wind_df, '-', color = 'tomato', lw = 0.7, label = 'Daily wind')
        ax2.set_ylabel("Wind velocity ($m.s^{-1}$)", color = 'tomato')
        ax2.tick_params(axis='y', colors='tomato')

        im = ax1.pcolormesh(df.index, df.columns, df.T,           # transpose DataFrame to have time on the x-axis
                shading="auto", cmap=cmap, vmin= vmini, vmax= vmaxi)
        ax1.set_ylabel("Diameter [nm]")
        ax1.set_xlabel("DateTime")
        ax1.set_title(main_title)
        # plt.colorbar(im, ax=ax1, pad=0.01)

        # Use make_axes_locatable to carve a fixed-width colorbar axis
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"Concentration ($cm^{-3}$)")

        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        
        
    def plot_hm(self, s:Literal['pos','neg', 'ratio']='pos', vmini = None, vmaxi = None, cmap = "RdBu_r"): #viridis?
        """Plot Q_snow and its components in an heat map"""

        if s == "pos":
            main_title = f"Positively charged particles ({self.low_dia} to {self.high_dia} nm)"
            data_dic = self.dic_pos
        elif s == "neg":
            main_title = f"Negatively charged particles ({self.low_dia} to {self.high_dia} nm)"
            data_dic = self.dic_neg
        elif s == "ratio":
            main_title = f"Positive / Negative ({self.low_dia} to {self.high_dia} nm)"
            df = self.dic_ratio
        else:
            raise ValueError("s must be 'pos', 'neg' or 'ratio'")
        
        nplots = len(data_dic)
        fig, axes = plt.subplots(
            nplots, 1, figsize=(15, 2.5 * nplots), sharex=True
        )

        if nplots == 1:
            axes = [axes]

        for ax, (title, df) in zip(axes, data_dic.items()):
        # transpose so: y = size, x = time
            im = ax.pcolormesh(df.index, df.columns, df.T,           # transpose DataFrame to have time on the x-axis
                shading="auto", cmap=cmap, vmin= vmini, vmax= vmaxi)

            ax.set_ylabel("Diameter [nm]")
            ax.set_title(title)
            # plt.colorbar(im, ax=ax, pad=0.01)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(r"Concentration ($cm^{-3}$)")

        axes[-1].set_xlabel("DateTime")
        fig.suptitle(main_title)
        fig.autofmt_xdate()
        plt.tight_layout()

    def plot_members(self, bin_ranges, s:Literal['pos','neg', 'ratio']='pos', commony :bool = False, logsc: bool = False):
        """Plot the contribution for each members of the Q_snow equation (sum of all bins)
        Inputs: - bin_ranges: list; Bin ranges used for plots [[x1,y1], [x2,y2], ...]
                - s: string; 'pos', 'neg' or 'ratio' (which is pos/neg)
                - commony: bool; If True, the y-axis scale is the same for all plot
                - logsc: bool; if True, y-axis is logarithmic"""

        if s == "pos":
            main_title = f"Positively charged particles"
            data_dic = self.dic_pos
            Q_snow = self.Q_snow_pos
        elif s == "neg":
            main_title = "Negatively charged particles"
            data_dic = self.dic_neg
            Q_snow = self.Q_snow_neg
        elif s == "ratio":
            main_title = f"Positive / Negative ({self.low_dia} to {self.high_dia} nm)"
            data_dic = self.dic_ratio
            Q_snow = self.dic_ratio[r"$Q_{\mathrm{snow}}$"]
        else:
            raise ValueError("s must be 'pos', 'neg' or 'ratio'")
        
        wind_df = self.met_df['true_wind_velocity']
        wind_mean = wind_df.mean()

        temp_mean = self.temperature_series.mean() if self.temperature is None else self.temperature
        
        nplots = len(bin_ranges)                #
        ncol = int(np.ceil(np.sqrt(nplots)))    # Design the subplot matrix
        nrow = int(np.ceil(nplots / ncol))               #
        fig, axs = plt.subplots(nrow,ncol, figsize = (ncol*6,nrow*4), sharex= True, sharey=commony, squeeze=False)

        for idx, (ax, (bin_low, bin_high)) in enumerate(zip(axs.flatten(), bin_ranges)):

            plotfun = ax.semilogy if logsc else ax.plot     # Decide whether log scale or not on y-axis
            col = idx%ncol  # column index for plotting columns

            ind_start = 2 if logsc else 1   # Plot dN/dt only if not log scale
            for lab, df in list(data_dic.items())[ind_start:]:
                plotfun(df.loc[:,bin_low:bin_high].sum(axis=1), alpha = 0.7, label = lab)

            # keep only pos values if log scale
            Q_snow_pos_values = Q_snow.loc[Q_snow.loc[:, bin_low:bin_high].sum(axis=1) > 0,bin_low:bin_high] if logsc else Q_snow.loc[:,bin_low:bin_high]
            plotfun(Q_snow_pos_values.sum(axis=1), color = "red", ls = '--', lw = 0.7, label = r"$Q_{\mathrm{snow}}$")

            if col == 0:
                ax.set_ylabel("Production rate [$cm^{-3}.s^{-1}$]")
            ax.set_xlabel("DateTime")
            ax.grid()
            subtitle = f"{bin_low} nm" if bin_low == bin_high else f"{bin_low} to {bin_high} nm"
            ax.set_title(subtitle)
            lines, labels = ax.get_legend_handles_labels()

            # plot wind
            color = "#d80ec7"
            ax2 = ax.twinx()
            ax2.plot(wind_df, color = color, alpha = 0.5, lw = 0.8, label = "Wind velocity")
            if col == ncol -1:
                ax2.set_ylabel("Wind ($m.s^{-1}$)", color=color) #, fontsize=8
                ax2.tick_params(axis='y', colors=color) #, labelsize=7
            else:
                ax2.tick_params(axis='y', colors=color, labelleft=False, labelright=False) #, labelsize=7

        fig.text(0.5, 0., rf"Average wind = {wind_mean:.2f} $m \cdot s^{{-1}}$" "\n" rf"Average temperature = {temp_mean:.2f} $K$", ha='center', va='center')
        fig.legend(lines, labels, loc = "upper center", ncol=len(data_dic))
        fig.suptitle(main_title)
        fig.autofmt_xdate()
        plt.tight_layout()

    def scatter_values(self, s = 'pos', x_data : Literal['wind', 'temperature', 'dtemp'] = 'wind',
                       bin_ranges = [(0.75, 31.62)], commony : bool = False,
                       ras: bool = False, pollution: bool = False):
        
        if x_data == 'wind':
            xvalues_raw = self.met_df['true_wind_velocity'].copy()
            xlab = f"Wind velocity ($m\\cdot s^{{-1}}$)"
        elif x_data == 'temperature':
            xvalues_raw = self.met_df['air_temperature'].copy()
            xlab = f"Temperature (°C)"
        elif x_data == 'dtemp':
            dtemp = self._diff(self.met_df[['air_temperature', 'air_pressure']])
            xvalues_raw = dtemp['air_temperature'].copy() * 6  # Multiply by 6 to have °C/h
            xlab = f"dT / dt (°C $\\cdot h^{{-1}}$)"
        
        if s=='pos':
            Q_snow = self.Q_snow_pos
            suptitle = "Positive ions"
        elif s=='neg':
            Q_snow = self.Q_snow_neg
            suptitle = "Negative Ions"
        else : raise ValueError("s must be 'pos' or 'neg'")

        x_values = xvalues_raw.reindex(Q_snow.index)
        
        # compute the masks according to the event type
        mask_event = self.event_tags == 'event'
        mask_poll = self.event_tags == 'event_poll'
        mask_ras = self.event_tags.isna()

        nplots = len(bin_ranges)                #
        ncol = int(np.ceil(np.sqrt(nplots)))    # Design the subplot matrix
        nrow = int(np.ceil(nplots / ncol))      #

        fig, axs = plt.subplots(nrow,ncol, figsize = (ncol*6,nrow*4), sharex= True, sharey=commony, squeeze=False)

        for idx, (ax, (bin_low, bin_high)) in enumerate(zip(axs.flatten(), bin_ranges)):
            
            Q_snow_sum = Q_snow.loc[:, bin_low:bin_high].sum(axis = 1)
            if ras == True:
                ax.scatter(x_values[mask_ras], Q_snow_sum[mask_ras], color = 'grey', alpha=0.4, s=10, label='no event')
            if pollution == True:
                ax.scatter(x_values[mask_poll], Q_snow_sum[mask_poll], color = 'tomato', alpha=0.6, s=15, label='polluted event')
            ax.scatter(x_values[mask_event], Q_snow_sum[mask_event], color = 'blue', alpha=.8, s=15, label='event')

            col = idx%ncol  # column index for plotting columns
            if col == 0:
                ax.set_ylabel("Production rate [$cm^{-3}.s^{-1}$]")
            if idx >= ncol * (nrow - 1):
                ax.set_xlabel(xlab)
            ax.grid()
            subtitle = f"{bin_low} nm" if bin_low == bin_high else f"{bin_low} to {bin_high} nm"
            ax.set_title(subtitle)
            lines, labels = ax.get_legend_handles_labels()
        fig.suptitle(suptitle)
        fig.legend(lines, labels, loc = "upper center", ncol=3)
        plt.tight_layout()
        
    def scatter_WT(self, s = 'pos',
                       bin_ranges = [(0.75, 31.62)], commony : bool = False,
                       ras: bool = False, pollution: bool = False):
        """Very similar with scatter_values, but scatter temperature against wind, with colors scaled on the production rate"""

        if s=='pos':
            Q_snow = self.Q_snow_pos
            suptitle = "Positive ions"
        elif s=='neg':
            Q_snow = self.Q_snow_neg
            suptitle = "Negative Ions"
        else : raise ValueError("s must be 'pos' or 'neg'")

        wind_raw = self.met_df['true_wind_velocity'].copy()
        wind = wind_raw.reindex(Q_snow.index)

        temp_raw = self.met_df['air_temperature'].copy()
        temp = temp_raw.reindex(Q_snow.index)
        xlab = "Temperature (°C)"

        # compute the masks according to the event type
        mask_event = self.event_tags == 'event'
        mask_poll = self.event_tags == 'event_poll'
        mask_ras = self.event_tags.isna()

        nplots = len(bin_ranges)                #
        ncol = int(np.ceil(np.sqrt(nplots)))    # Design the subplot matrix
        nrow = int(np.ceil(nplots / ncol))      #

        fig, axs = plt.subplots(nrow,ncol, figsize = (ncol*6,nrow*4), sharex= True, sharey=commony, squeeze=False)

        for idx, (ax, (bin_low, bin_high)) in enumerate(zip(axs.flatten(), bin_ranges)):
            
            Q_snow_sum = Q_snow.loc[:, bin_low:bin_high].sum(axis = 1)
            ax.scatter(temp[mask_event], wind[mask_event], c = Q_snow_sum[mask_event], alpha=.8, s=15, label='event')
        
            col = idx%ncol  # column index for plotting columns
            if col == 0:
                ax.set_ylabel("Wind [$m.s^{-1}$]")
            if idx >= ncol * (nrow - 1):
                ax.set_xlabel(xlab)
            ax.grid()
            subtitle = f"{bin_low} nm" if bin_low == bin_high else f"{bin_low} to {bin_high} nm"
            ax.set_title(subtitle)
            lines, labels = ax.get_legend_handles_labels()
        fig.suptitle(suptitle)
        fig.legend(lines, labels, loc = "upper center", ncol=3)
        plt.tight_layout()

    def scatter_3d(self, s = 'pos',
                       bin_ranges = [(0.75, 31.62)], commony : bool = False,
                       ras: bool = False, pollution: bool = False):
        
        if s=='pos':
            Q_snow = self.Q_snow_pos
            suptitle = "Positive ions"
        elif s=='neg':
            Q_snow = self.Q_snow_neg
            suptitle = "Negative Ions"
        else : raise ValueError("s must be 'pos' or 'neg'")

        wind_raw = self.met_df['true_wind_velocity'].copy()
        wind = wind_raw.reindex(Q_snow.index)

        temp_raw = self.met_df['air_temperature'].copy()
        temp = temp_raw.reindex(Q_snow.index)
        xlab = "Temperature (°C)"

        # compute the masks according to the event type
        mask_event = self.event_tags == 'event'
        mask_poll = self.event_tags == 'event_poll'
        mask_ras = self.event_tags.isna()

        nplots = len(bin_ranges)
        ncol = int(np.ceil(np.sqrt(nplots)))
        nrow = int(np.ceil(nplots / ncol))

        fig = plt.figure(figsize=(ncol * 6, nrow * 5))

        for idx, (bin_low, bin_high) in enumerate(bin_ranges):

            ax = fig.add_subplot(nrow, ncol, idx + 1, projection='3d')
            Q_sum = Q_snow.loc[:, bin_low:bin_high].sum(axis=1)

            ax.scatter(wind[mask_event], temp[mask_event], Q_sum[mask_event],
                        color='blue', alpha=0.8, s=15, label='event')
            
            ax.set_xlabel("Wind ($m\\cdot s^{-1}$)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_zlabel("Production rate [$cm^{-3}\\cdot s^{-1}$]")
            subtitle = f"{bin_low} nm" if bin_low == bin_high else f"{bin_low} to {bin_high} nm"
            ax.set_title(subtitle)
        
        handles, labels = ax.get_legend_handles_labels()
        fig.suptitle(suptitle)
        fig.legend(handles, labels, loc="upper center", ncol=3)
        plt.tight_layout()