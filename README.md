# Fire Station Optimisation via Genetic Algorithm

This repository presents a spatial optimisation framework for fire station placement using a Genetic Algorithm (GA). It combines machine learning (Random Forest) and network-based travel time estimation (via OSRM) to evaluate and optimise fire service efficiency at the grid level.


## Project Structure

```text
Fire_Station_Optimisation_GA/
├── optimiser/                    # Core optimisation logic
│   ├── GAOptimiser.py            # Genetic Algorithm 
│   ├── ga_run.py                 # Run GA
│   ├── Output                    # outputs
│   └── log                       # log files
│ 
├── feasibility_study/            # Feasibility study
│   ├── methods/                  # Functions for feasibility study
│   │   ├── best_k.py             # Based on best_K selections
│   │   └── demand_weighted.py    # Demand weighted single sample
│   └── run.py                    # Run the feasibility study
│ 
├── uilts/                       # common tools for GA and feasibility study 
│   ├── config.py                 # GA hyperparameters
│   ├── data_loader.py            # Load model input data into global variables
│   ├── validation.py             # Validation
│   └── evaluator.py              # Calculate the fitness 
│ 
├── data/                         # Input data (e.g., .npy, .joblib, .gpkg files and time matrix)
│   └── ... 
├── results/                      # Results
│   ├── best_k
│   └── demand_weighted
│ 
├──  README.md                    # This file 
├── .environment.yml              # The environment of this repo
├── .gitignore 
└── .gitattributes

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
