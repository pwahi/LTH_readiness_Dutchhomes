## Instructions for analysing the effect of oversizing on lower temperature readiness.

This folder contains a Jupyter notebook and data used for analysing the effect of radiator oversizing on the readiness of the representative samples 
for `Medium Temperature` (70/50&deg;C) and `Low Temperature` (55/35&deg;C). 
## Contents

1. **Jupyter Notebook**
   - `TH_radiator_oversizing.ipynb`: Jupyter notebook to analyse effect of oversizing on lower temperature readiness.
2. **Data_Directory** : `Data_generic_ovs` : Simulation results for identified sample size of 1300 under High, Medium and Low Temperature supply under different radiator oversizing factors. 
   - `Data_generic_ovs` : Simulated outputs with **_generic oversizing factor_**.
   - `Data_ovs_1pt25` : Simulated outputs with **_factor of 1.25_**.
   - `Data_ovs_1pt66` : Simulated outputs with **_factor of 1.66_**.
   - `Data_ovs_2pt5` : Simulated outputs with **_factor of 2.5_**.
   - `Data_ovs_5` : Simulated outputs with **_factor of 5_**.

## Dependencies

Ensure you have the following Python libraries installed:

- pandas
- pathlib

## Instructions

The jupyter notebook illustrates the steps for data processing and evaluating effect of radiator oversize factors. 