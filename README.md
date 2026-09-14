# Aerosol-size-distribution
Analysis of the ion size distribution and formation rate during blowing snow events, using data collected during the MOSAiC expedition (2019/2020). This work is a part my master's degree's first year's internship (STPE-SCAHC master's degree, Université Grenoble Alpes), with Hans-Werner JACOBI as supervisor.

## Project structure
|-- travail.py 				# Main Script

|-- exe_all.py				# Global result script

|-- preprocess.py			# CSV pre-processing

|-- ion-formation-rate3		# IonFormation3 class

|-- requirements.txt		# Python dependencies

|-- Data/					# Raw data (not tracked by git)

	-- days_of_interest_notes_20240214

	-- nais_neg_ions_raw

	-- nais_pos_ions_raw

	-- nais_neg_particles_raw

	-- polarstern_weather

	-- smps_psd_5min_raw

|-- Data-clean/				# Clean data (wrote by preprocess.py) (not tracked by git)

└-- Results/				# Resulting plots and csv

## Installation
> git clone https://github.com/malonoiret-cpu/Aerosol-size-distribution.git

> cd Aerosol-size-distribution

> python -m venv venv

> venv\Scripts\activate

> pip install -r requirements.txt

## Usage
- Place raw data files in the 'Data/' folder
- Run preprocess.py to generate clean parquet files
- Run (and/or edit) travail.py to compute ion formation rate and generate plots
- Run exe_all to save plots for all events according to the plot functions called in the script

## Acknowledgment
With sincere thanks to:
- Hans-Werner Jacobi (Université Grenoble Alpes, Institut des géosciences de l'environnement) for his guidance and supervision
- Kateryna Tkachenko (National Academy of Science of Ukraine) and Matthew Boyer (University of Helsinki) for their help and feedback on my results.

