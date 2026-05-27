# this class is based on ion_formation_rate2.py, and aim to give results according to the size bins

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from typing import Literal

class IonFormation:
    def __init__(self, particle_psd: pd.DataFrame, pos_ion_psd: pd.DataFrame, neg_ion_psd: pd.DataFrame, low_dia=None, high_dia=None, \
			  		pressure = 101.3, temperature = 298., alpha = 1.6e-6, chi = 0.01e-6, rho = 0.00183):
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

        ## Met data? Shall we consider DataFrames to have the temperature and pressure for each date?
        self.pressure = pressure
        self.temperature = temperature
        
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
        self.nucmode_pos_ion_psd = self.pos_ion_psd.loc[:, self.low_dia:self.high_dia] # slice the size distribution down to the selected size bin of interest
        self.pos_N_ion = self.nucmode_pos_ion_psd # [#/cm**3], the concentration of pos ions between dp_min and dp_max

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
        self.dtime = np.diff(self.nucmode_pos_ion_psd.index).astype(float) /1e9             # [s], divide by 1e9 to convert nanoseconds to seconds
        self.dtime = self.dtime[:, None]
            ## Compute the members of the Q_snow_pos equation                                            
        self.dNdp_pos_ion = pd.DataFrame(                                                               # calculate the change in the nucmode_pnc over time
                np.diff(self.pos_N_ion.values, axis=0),
                index=self.pos_N_ion.index[:-1],
                columns=self.pos_N_ion.columns
            )
        self.pos_coag_loss_term = self.calc_coag_loss(ion_psd = self.pos_ion_psd)[:-1] * self.pos_N_ion[:-1]
        self.pos_growth_rate_term = 0
        self.pos_alpha_term = self.alpha * self.pos_N_ion[:-1] * self.N_neg_ion_smaller[:-1]
        self.pos_chi_term = self.chi * self.N_particle[:-1] * self.N_pos_ion_smaller[:-1]

            ## Compute the members of the Q_snow_neg equation 
        
        self.dNdp_neg_ion = pd.DataFrame(                                                               # calculate the change in the nucmode_pnc over time
                np.diff(self.neg_N_ion.values, axis=0), # returns numpy array, so it's needed to make a DataFrame
                index=self.neg_N_ion.index[:-1],
                columns=self.neg_N_ion.columns
            )
        self.neg_coag_loss_term = self.calc_coag_loss(ion_psd = self.neg_ion_psd)[:-1] * self.neg_N_ion[:-1]
        self.neg_growth_rate_term = 0
        self.neg_alpha_term = self.alpha * self.neg_N_ion[:-1] * self.N_pos_ion_smaller[:-1]
        self.neg_chi_term = self.chi * self.N_particle[:-1] * self.N_neg_ion_smaller[:-1]
        
        self.Q_snow_pos = self.Q_snow_calc(s = 'pos')
        self.Q_snow_neg = self.Q_snow_calc(s = 'neg')
        ## -------------------------------------------------------------------------------------------------------------------------------

        # store the results in dic for plots
        self.dic_pos = {
                r"$Q_{\mathrm{snow}}$": self.Q_snow_pos,
                r"$\partial N / \partial t$": self.dNdp_pos_ion / self.dtime,
                r"Coagulation loss": self.pos_coag_loss_term,
                r"$\alpha$ term": self.pos_alpha_term,
                r"$\chi$ term": self.pos_chi_term,
            }
        self.dic_neg = {
                r"$Q_{\mathrm{snow}}$": self.Q_snow_neg,
                r"$\partial N / \partial t$": self.dNdp_neg_ion / self.dtime,
                r"Coagulation loss": self.neg_coag_loss_term,
                r"$\alpha$ term": self.neg_alpha_term,
                r"$\chi$ term": self.neg_chi_term,
            }
        


    def N_smaller(self, psd):
        """Compute the number of smaller particle than a bin for each bin size (cumsum)
        Input: pos or neg ion psd"""
        full_cumsum = psd.cumsum(axis = 1)
        full_cumsum_shifted = full_cumsum.shift(1, axis=1).fillna(0)    # The first bin is filled with 0s, the second with the concentration of the first bin, the third the sum of the two first...
        return full_cumsum_shifted

    def mean_free_path_calc(self):
        """Compute the mean free path from the reference (at T = 296.15 [K], P = 101.3 [kPa]; MFP_REF = 6.730e-8 [m])
        Output: float: mean free path [m]"""
        mfp = self.MFP_REF * (self.P_REF / self.pressure) * (self.temperature / self.T_REF)                \
            * ((1.0 + self.SUTHERLAND / self.T_REF)/(1.0 + self.SUTHERLAND / self.temperature)) # [m])
        return mfp
	
    def viscosity_calc(self):
        """Compute the viscosity (simplification of the TSI equation, given by Ernie Lewis)
        Input: temperature [K] (default 298 K)
        Output: float: viscosity"""
        return self.VISCOSITY_REF * (self.temperature / self.T_REF)**0.78

    def calc_coag_loss(self, ion_psd):
        """Compute the coagulation loss. First, all units are converted to centimeters to compute CoagS in cm. Then, the total coag loss for each time step is calculated.
        Inputs:	- the pos or neg ion psd
        Outputs:	- array containing all total CoagL for each time step
                        -> result differs to coag_loss_1 with at the 10^-11th decimal"""

        # convert all dimensions for the coagulation losses to cm
        boltzmann = self.BOLTZMANN * 10000 		# units for boltzmann constant is in [m**2 kg/s**2 K], multiplied by 10000 to convert to [cm**2*kg/s**2*K]
        mfp = self.mean_free_path_calc() * 100 	# mean free path multiplied by 100 to get it in [cm]
        mu = self.viscosity_calc() / 100 		# viscosity, divide by 100 to convert [kg/m*s] to [kg/cm*s]

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

        ## Compute values for Kij
        # slip correction factor
        Cc1 = 1.0 + (2.0 * mfp / d1) * (1.257 + 0.4 * np.exp(-1.1 * d1 / (2.0 * mfp)))  # (N, 1)
        Cc2 = 1.0 + (2.0 * mfp / d2) * (1.257 + 0.4 * np.exp(-1.1 * d2 / (2.0 * mfp)))  # (1, M)

        # diffusivity [cm**2/s]
        D1 = (boltzmann * self.temperature * Cc1) / (3.0 * np.pi * mu * d1)  # (N, 1)
        D2 = (boltzmann * self.temperature * Cc2) / (3.0 * np.pi * mu * d2)  # (1, M)

        # mass of particles [kg]
        m1 = (1.0 / 6.0) * np.pi * (d1**3.0) * self.rho		# (N, 1)
        m2 = (1.0 / 6.0) * np.pi * (d2**3.0) * self.rho		# (1, M)

        # average velocity [cm/s]
        c_bar1 = ((8.0 * boltzmann * self.temperature) / (np.pi * m1))**0.5  # (N, 1)
        c_bar2 = ((8.0 * boltzmann * self.temperature) / (np.pi * m2))**0.5  # (1, M)

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
        
        # upper triangle mask: keeps only j >= i pairs, zeros the rest (replaces j loop range(i, M))
        triu_mask = np.triu(np.ones((N, M), dtype=bool))  # shape (N, M)

        # self-coagulation mask: diagonal where i == j
        diag_idx = np.arange(min(N, M)) # diagonal indices to apply the mask later

        # loop through each SMPS scan (every timestamp) associated with the NPF event
        coag_loss_all_sum = []	# list to save the cumulative coagulation loss in the nuc mode for each scan (final list)

        for t in range(len(ion_psd.index)):
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
        Q_snow_pos = self.dNdp_pos_ion / self.dtime \
                    + self.pos_coag_loss_term   \
                    + self.pos_growth_rate_term \
                    + self.pos_alpha_term       \
                    - self.pos_chi_term
        
        Q_snow_neg = self.dNdp_neg_ion / self.dtime \
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
        
    def rollmean(self, df = pd.DataFrame(), T = '30min'):
        return df.rolling(window=T, center = True).mean()
        
    def Q_snow_plot(self, s: Literal['all', 'pos', 'neg'] = "all", T = '30min'):
        plt.figure(figsize=(10, 6))

        if s=='all' or s=='pos':
            Q_snow = np.sum(self.Q_snow_pos, axis=1)
            plt.plot(Q_snow, '.', color='tomato', alpha = 0.5, label='_Q snow pos')
            plt.plot(self.rollmean(Q_snow), color = 'tomato', label = 'Q snow pos')

        if s=='all' or s=='neg':
            Q_snow = np.sum(self.Q_snow_neg, axis=1)
            plt.plot(Q_snow, '.', color='blue', alpha = 0.5, label='_Q snow neg')
            plt.plot(self.rollmean(Q_snow), color = 'blue', label = 'Q snow neg')
        # Set the format for the date on the axis X
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # Формат дати (date format)
        # plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Визначаємо інтервал для підписів (кожні 6 годин) (6 hours interval signature)
        plt.xticks(rotation=45)

        plt.xlabel('Time')
        plt.ylabel('Q snow [#/cm^3/s]')
        plt.title('Q snow vs Time')
        plt.legend()
        plt.grid()
        plt.tight_layout()
    
    # def all_plot(conc_df, met_df, T='72h', bin_ranges = bin_ranges):
	#     """Print the global concentration of particles"""

        

    def plot_hm(self, s:Literal['pos','neg']='pos', vmini = None, vmaxi = None, cmap = "RdBu_r"): #viridis?
        """Plot Q_snow and its components in an heat map"""

        if s == "pos":
            main_title = f"Positively charged particles ({self.low_dia} to {self.high_dia} nm)"
            data_dic = self.dic_pos
        elif s == "neg":
            main_title = f"Negatively charged particles ({self.low_dia} to {self.high_dia} nm)"
            data_dic = self.dic_neg
        else:
            raise ValueError("s must be 'pos' or 'neg'")
        
        nplots = len(data_dic)
        fig, axes = plt.subplots(
            nplots, 1, figsize=(15, 2.5 * nplots), sharex=True
        )

        if nplots == 1:
            axes = [axes]

        for ax, (title, df) in zip(axes, data_dic.items()):
        # transpose so: y = size, x = time
            im = ax.pcolormesh(
                df.index,
                df.columns,
                df.T,           # transpose DataFrame to have time on the x-axis
                shading="auto",
                cmap=cmap,
                vmin= vmini,
                vmax= vmaxi,
            )

            ax.set_ylabel("Diameter [nm]")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, pad=0.01)
        
        axes[-1].set_xlabel("Time")
        fig.suptitle(main_title)
        fig.autofmt_xdate()
        plt.tight_layout()

    def plot_members(self, s:Literal['pos','neg']='pos', bin_ranges = [[.75,5.62], [10.,31.62]]):
        """Plot the contribution for each members of the Q_snow equation (sum of all bins)"""

        if s == "pos":
            main_title = f"Positively charged particles"
            data_dic = self.dic_pos
            Q_snow = self.Q_snow_pos
        elif s == "neg":
            main_title = "Negatively charged particles"
            data_dic = self.dic_neg
            Q_snow = self.Q_snow_neg
        else:
            raise ValueError("s must be 'pos' or 'neg'")
        
        fig, axs = plt.subplots(1,2, figsize = (10,6))
        
        for ax, (bin_low, bin_high) in zip(axs, bin_ranges):
            ax.plot(Q_snow.loc[:,bin_low:bin_high].sum(axis=1), color = "red", label = "Q_snow")
            for lab, df in list(data_dic.items())[2:]:
                ax.plot(df.loc[:,bin_low:bin_high].sum(axis=1), alpha = 0.5, label = lab)

            ax.set_xlabel("DateTime")
            ax.set_ylabel(r"Production rate \[$cm^{-3}.s^{-1}$\]")
            ax.legend()
            ax.set_title(f"{bin_low} to {bin_high} nm")
        fig.suptitle(main_title)
        plt.grid()
        fig.autofmt_xdate()
        plt.tight_layout()

    
    