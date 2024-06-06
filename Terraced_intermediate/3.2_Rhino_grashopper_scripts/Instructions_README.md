# Instructions for parametric simulations

This folder contains scripts and resources related to simulating the samples representing terraced-intermediate type
parametrically for different supply temperature. 

## Contents

1. **Rhino-Grasshopper files**
   - Terraced_house.3dm : Rhino file with seed model
   - Terraced_house.gh : Accompanying grasshopper script
2. **Weather files**
   - NEN5060_min10_Hddy.ddy : DDY file for heat loss calculations
   - PW_NEN5060_2021_EPW.epw : NEN5060 TRY file for simulations.
3. **Python scripts**
   - hops_calculation_script.py : Python script for hops component in GH script.
4. **Jupyter notebooks**
   - Pollination_API.ipynb : Jupyter notebook to use Pollination API to simulate HBjson files generated from the GH script. 
5. **Directories**
   - These directories are needed to store various files required for simulations. 
     1. Input_excel : Samples in Excel format, used as an input by GH files to generate HBjson files. 
     2. Output_excel : Excel file to record the output of simulations.
     3. HBJson : To store the HBjson of every sample from GH script
     4. Sim_par : To store the corresponding simulation parameters for every HBjson from GH script.
     5. Sql_files : To download the simulation results from Pollination API.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Rhino**: Version 7.0. [Download Rhino](https://www.rhino3d.com/download)
- **Grasshopper**: Built into Rhino, ensure it is installed and updated.
- **GH plugins** :
  - Ladybug Honeybee v1.6
  - Metahopper v1.2.4
  - TTTools v2.0.3
  - Hops v0.16.2.0
- **Python libraries**
  - os
  - pathlib
  - sqlite3
  - pandas
  - flask
  - ghhops_server
  - scipy

## Instructions

Follow these steps to use parametrically simulate

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/your-repository.git
