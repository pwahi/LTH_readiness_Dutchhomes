# Instructions for Parametric Simulations

This folder contains scripts, resources, and instructions for simulating samples representing terraced-intermediate types parametrically for different supply temperatures.

## Contents

1. **Rhino-Grasshopper Files**
   - `Terraced_house.3dm`: Rhino file with seed model.
   - `Terraced_house.gh`: Accompanying Grasshopper script.

2. **Weather Files**
   - `NEN5060_min10_Hddy.ddy`: DDY file for heat loss calculations.
   - `PW_NEN5060_2021_EPW.epw`: NEN5060 TRY file for simulations.

3. **Python Scripts**
   - `hops_calculation_script.py`: Python script for Hops component in the Grasshopper script.

4. **Jupyter Notebooks**
   - `Pollination_API.ipynb`: Jupyter notebook to use Pollination API to simulate HBjson files generated from the Grasshopper script.

5. **Directories**
   - These directories store various files required for simulations:
     1. `Input_excel`: Contains samples in Excel format, used as input by Grasshopper files to generate HBjson files.
     2. `Output_excel`: Stores Excel files to record the output of simulations.
     3. `HBJson`: Stores the HBjson of every sample from the Grasshopper script.
     4. `Sim_par`: Stores the corresponding simulation parameters for every HBjson from the Grasshopper script.
     5. `Sql_files`: Stores the simulation results downloaded from the Pollination API.

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

- The Grasshopper script uses the samples to create HBjson files for simulations. Therefore, before using the scripts, ensure you have the input Excel file exported from the sampling script (see section 3.3 / directory 3.3.1 sampling scripts) in the `Input_excel` subdirectory.
- Make a copy of the exported input file and save it in `Output_excel`. The simulation results will be written to the last two columns of this Excel file.

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
