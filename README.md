# Aerosol-size-distribution
Analysis of the ion size distribution and formation rate during blowing snow events, using data collected during the MOSAiC expedition (2019/2020). This work has been done during my master's degree's first year's internship, with Hans-Werner JACOBI as supervisor.

## Project structure
|-- travail.py 				# Main Script

|-- preprocess.py			# CSV pre-processing

|-- ion-formation-rate3		# IonFormation3 class

|-- requirements.txt		# Python dependencies

|-- Data/					# Raw data (not tracked by git)

|-- Data-clean/				# Clean data (wrote by preprocess.py) (not tracked by git)

└-- Results/				# Resulting csv (and parquet?) after calculation

## Installation
git clone https://github.com/malonoiret-cpu/Aerosol-size-distribution.git

cd Aerosol-size-distribution (first two on the same line ?)

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

## Usage
- Place raw data files (see below) in the 'Data/' folder
- Run preprocess.py to generate clean parquet files (see below)
- Run (and/or edit) travail.py to compute ion formation rate and generate plots

## Data files
This project has been made with csv data file as inputs. The csv were made as follow:
- column headers: time,$X_1$,$X_2$,...,$X_i$ ($X_i$ beeing the particle size bins with $X_1$ the smallest, in nanometer)
- time column: yyyy-mm-dd hh:mm:ss ('latin-1')
- particle size columns: dndlogdp

## Preprocess
"Clean the data" means converting the units from dndlogdp to concentration, setting the time column as a DateTime Index, and converting the column headers to numeric. The data are saved as parquet files so they are ready to import and use in travail.py.

## Bibliography
...
