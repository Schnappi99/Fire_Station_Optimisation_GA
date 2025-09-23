# Fire Station Optimisation via Genetic Algorithm

This repository presents a spatial optimisation framework for fire station placement using a Genetic Algorithm (GA). It combines machine learning (Random Forest) and network-based travel time estimation (via OSRM) to evaluate and optimise fire service efficiency at the grid level.


## Project Structure

```text
Fire_Station_Optimisation_GA/
├── main.py                       # Entry point to run GA optimisation
├── optimiser/                    # Core optimisation logic
│   ├── GA_algorithm.py           # Genetic Algorithm and fitness function
│   ├── data_loader.py            # Load model input data into global variables
│   └── config.py                 # GA hyperparameters
│   └── feasiblilty_study.py      # Feasibility study
│   └── ga_runner.py              # Run GA
│ 
├── utils/                        # Helper scripts and notebooks
│   ├── osrm_utils.py             # Functions for computing OSRM travel time
│   └── preprocess.ipynb          # Grid/feature generation and preprocessing
│   └── osrm_drv_time.ipynb       # Local attempts to calculate time for osrm
│   └── driving_time_matrix.py    # Calculate the osrm driving time matrix
│ 
├── analysis/                     # Evaluation tools
│   ├── feasibility_study.py      # Compare current layout to random layouts
│   ├── random_layouts.csv        # Results of 1000 random layouts
│   └── random_layout_hist.png    # Histogram of fitness distribution
│
├── data/                         # Input data (e.g., .npy, .joblib, .gpkg files)
│   └── ... 
└── README.md                     # This file

```

## Getting starts
```bash
# Clone the repository
git clone https://github.com/Schnappi99/Fire_Station_Optimisation_GA.git
cd Fire_Station_Optimisation_GA

# Install dependencies
conda env create -f environment.yml
conda activate fire_station_ga

# Run feasibility benchmark
python optimiser/feasibility_study_weighted.py

# Run optimisation
python optimiser/ga_runner.py

```
