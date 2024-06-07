# Instructions for Parametric Simulations

This folder contains scripts, resources, and instructions for simulating terraced-intermediate samples parametrically for different supply temperatures.
![Simulation_workflow](Assets/simulation%20workflow.tif)

## Contents

1. **Rhino-Grasshopper Files**
   - `Terraced_house.3dm`: Rhino file with the seed model.
   - `Terraced_house.gh`: Accompanying Grasshopper script.

2. **Weather Files**
   - `NEN5060_min10_Hddy.ddy`: DDY file for heat loss calculations.
   - `PW_NEN5060_2021_EPW.epw`: NEN5060 TRY file for simulations.

3. **Python Scripts**
   - `hops_calculation_script.py`: Python script for Hops component in the Grasshopper script.

4. **Jupyter Notebooks**
   - `Pollination_API.ipynb`: Jupyter notebook to use Pollination API for simulating HBjson files generated from the Grasshopper script.

5. **Directories**
   - These directories store various files required for simulations:
     - `Input_excel`: Contains samples in Excel format, used as input by Grasshopper files to generate HBjson files.
     - `Output_excel`: Stores Excel files to record the output of simulations.
     - `HBJson`: Stores the HBjson files of every sample from the Grasshopper script.
     - `Sim_par`: Stores the corresponding simulation parameters for every HBjson from the Grasshopper script.
     - `Sql_files`: Stores the simulation results downloaded from the Pollination API.

## Note

Separate subdirectories should be created inside `HBJson`, `Sim_par`, and `Sql_files` for each iteration run. The naming convention for the subdirectories should follow: `date_hbjson/simpar/sqlfiles_itr_(iteration number)_size_(size of the sample)_(supply temperature: HT/MT/LT)`. This is important for the `Pollination_API.ipynb` notebook to find the correct HBjson and simulation parameter files and download the SQL files to the correct folder.

For example:

```
1. Input_excel
│ └── 2024-04-23_inputfile_itr_1_size_1300.xlsx
2. Output_excel
│ └── 2024-04-23_outputfile_itr_1_size_1300_HT.xlsx
3. HBjson
│ └── 2024-04-23_hbjson_itr_1_size_1300_HT
│ ├── Th_hbjson_sample_1.hbjson
│ ├── Th_hbjson_sample_2.hbjson
│ └── ...
4. Sim_par
│ └── 2024-04-23_simpar_itr_1_size_1300_HT
│ ├── Th_simpar_sample_1.json
│ ├── Th_simpar_sample_2.json
│ └── ...
5. Sql_files
└── 2024-04-23_sqlfiles_itr_1_size_1300_HT
├── Th_hbjson_sample_1.sql
├── Th_hbjson_sample_2.sql
└── ...
```
The `TH_sampling.ipynb` jupyter notebook can be used to sample as well as create these subdirectories in a dedicated project folder.
Follow instructions in [3.3.1_Sampling_scripts](https://github.com/pwahi/LTH_readiness_Dutchhomes/tree/main/Terraced_intermediate/3.3.1_Sampling_scripts)
    
## Dependencies

Before you begin, ensure you have met the following requirements:

- **Rhino**: Version 7.0.
- **Grasshopper**: Built into Rhino, ensure it is installed and updated.
- **Grasshopper Plugins**:
  - Ladybug Honeybee v1.6
  - Metahopper v1.2.4
  - TTTools v2.0.3
  - Hops v0.16.2.0
- **Python Libraries**:
  - os
  - pathlib
  - sqlite3
  - pandas
  - flask
  - ghhops_server
  - scipy
  - time
  - requests
  - zipfile
  - tempfile
  - shutil
  - glob
  - datetime
  - typing
  - pollination_streamlit
  - queenbee

## Instructions

### Prerequisite

- The Grasshopper script uses the samples to create HBjson files for simulations. Therefore, before using the scripts, ensure you have the input Excel file exported from the sampling script ([3.3.1 sampling scripts](https://github.com/pwahi/LTH_readiness_Dutchhomes/tree/main/Terraced_intermediate/3.3.1_Sampling_scripts)) in the `Input_excel` subdirectory.
- Make a copy of the exported input file and save it in `Output_excel`. The simulation results will be written to the last two columns of this Excel file.
- You need a working API licence to use [Pollination cloud computing](https://www.pollination.cloud/). 
### Steps

1. **Open `Terraced_house.3dm` in Rhino.**
2. **Run `hops_calculation_script.py` before opening the Grasshopper script.**
3. **Follow the steps in the Grasshopper script:**
   - **Step 1-4**: Select the correct input and output Excel files and subdirectories to store HBjson files.
   - **Step 5**: Ensure the component reads every cell of the input Excel file.
   - **Step 6**: Update the iterator slider to the number of samples from the panel at step 5.
   - **Step 7**: Choose the supply temperature level for which the samples must be simulated.
   - **Step 8/9**: Toggle the Boolean to `True` to write HBjson and Sim_par Json files.
   - **Step 10**: Clean the GBO recorder before step 11.
   - **Step 11**: Click "fly" to iterate through all the samples in the input Excel file and generate HBjson and Sim_par Json files.

4. **Simulate Models Using Pollination API**:
   - Use the Jupyter notebook `Pollination_API.ipynb` to upload and simulate the models on the Pollination server. The simulation results (SQL files) will be downloaded to the `Sql_files` subdirectory.

5. **Process Simulation Results**:
   - Once the SQL files are downloaded, follow the steps in the Grasshopper script:
     - **Step 12**: Select all SQL files and toggle `True` to read them using the Hops server.
     - **Step 13**: Toggle `True` to write the results in the `Output_excel`.

By following these steps, you can ensure a smooth and effective simulation process for the terraced-intermediate type samples at different supply temperatures.

References:
The code for Pollination API is developed using this discussion on the [Pollination forum](https://discourse.pollination.cloud/t/unable-to-load-a-large-number-of-runs-as-one-job/1446/9?u=prateekwahi)
