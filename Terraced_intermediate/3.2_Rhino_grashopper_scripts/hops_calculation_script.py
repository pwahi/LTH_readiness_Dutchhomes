# Import libraries
import os
import pathlib
import sqlite3
import pandas as pd
from flask import Flask
import ghhops_server as hs
from scipy.optimize import fsolve
import numpy as np

# Create a Flask application
app = Flask(__name__)
# Initialise Hops Server
hops = hs.Hops(app)

# Decorators are made to register a new component with the Hops server.
# This component will be called remotely in the Grasshopper Script.
# This was done to expand the functionality of Python in Grasshopper.
# It is expected that Python can be used with other libraries in Rhino v8.


# Register a new HOPS component with endpoint "/heating_kwh"
@hops.component(
    "/heating_kwh",
    name="Heating kWh",
    description="Get heating kwh from SQL file",
    inputs=[
        hs.HopsString("SQL Path", "Path", "Paths to SQL file"),
        hs.HopsString("Zone names", "Zone_names", "Comma-Separated list of zone names"),
    ],
    outputs=[
        hs.HopsNumber("Heating kWh", "H", "Heating kWh"),
    ]
)
def heating_kwh(sql_path, zone_names):
    """
        Function to calculate heating energy in kWh for specified zones from an EnergyPlus SQL file.

        sql_path: Path to the EnergyPlus SQL file.
        zone_names: Comma-separated list of zone names.
        return: Total heating energy in kWh.
    """

    abs_sql_path = os.path.abspath(sql_path) # Get the absolute path of the SQL file
    sql_uri = '{}?mode=ro'.format(pathlib.Path(abs_sql_path).as_uri()) # Create URI for the SQL file

    query = 'SELECT EnergyPlusVersion FROM Simulations'
    with sqlite3.connect(sql_uri, uri=True) as con: # Connect to the SQLite database in read-only mode
        cursor = con.cursor()
        r = cursor.execute(query).fetchone()
        if r:
            simulation_info = r[0] # Retrieve EnergyPlus version information
        else:
            msg = ("Cannot find the EnergyPlusVersion in the SQL file. "
                   "Please inspect query used:\n{}".format(query))
            raise ValueError(msg) # Raise an error if EnergyPlus version is not found

    zone_names = zone_names.split(',') # Split the zone names into a list
    variables = ['Zone Ideal Loads Supply Air Total Heating Energy']
    df_all_zones = pd.DataFrame()

    for zone_name in zone_names:
        # Query to get the index values for each zone and variable
        index_values = [pd.read_sql(
            "SELECT ReportVariableDataDictionaryIndex FROM ReportVariableDataDictionary WHERE KeyValue LIKE '%{}%' AND VariableName = '{}'".format(
                zone_name, variable), con=con) for variable in variables]
        index_values = [item for sublist in index_values for item in
                        sublist['ReportVariableDataDictionaryIndex'].tolist()]

        if index_values:
            # Query to get the report data for each zone based on the index values
            df_zone_data = pd.read_sql("SELECT * FROM ReportData WHERE ReportDataDictionaryIndex IN ({})".format(
                ','.join(str(index) for index in index_values)), con=con)
            df_zone_data['Zone'] = zone_name
            df_all_zones = pd.concat([df_all_zones, df_zone_data], ignore_index=True)

    sum_of_values = df_all_zones['Value'].sum() # Sum the values for all zones
    heating_kwh = sum_of_values * 2.7777778e-7 # Convert the sum to kWh

    return heating_kwh # Return the total heating energy in kWh


@hops.component(
    "/temperature",
    name="Temperature",
    description="Get hourly operative temperature from SQL file",
    inputs=[
        hs.HopsString("SQL Path", "Path", "Paths to SQL file"),
        hs.HopsString("Temperature zone name", "Temp_zone_name", "Name of the zone for temperature calculations")
    ],
    outputs=[
        hs.HopsString("Temperature Data", "T", "Hourly operative temperature data for the zone"),
    ]
)
def temperature(sql_path, temp_zone_name):
    """
    Function to get hourly operative temperature data for a specified zone from an EnergyPlus SQL file.

    sql_path: Path to the EnergyPlus SQL file.
    temp_zone_name: Name of the zone for temperature calculations.
    return: Hourly operative temperature data in JSON format.
    """

    abs_sql_path = os.path.abspath(sql_path) # Get the absolute path of the SQL file
    sql_uri = '{}?mode=ro'.format(pathlib.Path(abs_sql_path).as_uri()) # Create URI for SQLite connection

    query = 'SELECT EnergyPlusVersion FROM Simulations'
    with sqlite3.connect(sql_uri, uri=True) as con: # Connect to the SQLite database in read-only mode
        cursor = con.cursor()
        r = cursor.execute(query).fetchone()
        if r:
            simulation_info = r[0]
        else:
            msg = ("Cannot find the EnergyPlusVersion in the SQL file. "
                   "Please inspect query used:\n{}".format(query))
            raise ValueError(msg) # Retrieve EnergyPlus version information

    variables = ['Zone Operative Temperature']
    df_zone_data = pd.DataFrame()

    # Query to get the index values for the specified zone and variable
    index_values = [pd.read_sql(
        "SELECT ReportVariableDataDictionaryIndex FROM ReportVariableDataDictionary WHERE KeyValue LIKE '%{}%' AND VariableName = '{}'".format(
            temp_zone_name, variable), con=con) for variable in variables]

    index_values = [item for sublist in index_values for item in sublist['ReportVariableDataDictionaryIndex'].tolist()]

    if index_values:
        # Query to get the report data for the specified zone based on the index values
        df_zone_data = pd.read_sql("SELECT * FROM ReportData WHERE ReportDataDictionaryIndex IN ({})".format(
            ','.join(str(index) for index in index_values)), con=con)
        df_zone_data['Zone'] = temp_zone_name

    temperature_data = df_zone_data['Value'].to_json(orient='records') # Convert the temperature data to JSON format

    return temperature_data # Return the temperature data in JSON format


@hops.component(
    "/compact_ratio",
    name="Compactness ratio",
    description="Calculate the corresponding length of the house required to achieve the compactness ratio "
                "from the sample. ",
    inputs=[
        hs.HopsNumber("Compact_ratio", "CR", "Cr for calculating Length"),
    ],
    outputs=[
        hs.HopsNumber("L", "L", "Solution for L"),
    ]
)
def solve(CR_given):
    """
        Function to calculate the length of the house required to achieve the given compactness ratio.

        CR_given: Compactness ratio.
        return: Length of the house.
    """
    def equation(L, CR):
        """
        Equation to solve for the length L based on the given compactness ratio CR.
        """
        return ((3.89 * L) + 61.56 + (10.8 * np.sqrt(24.01 + (L / 2) ** 2))) / (14.58 * L) - CR

    L_initial_guess = 1  # Initial guess for the length
    L_solution = fsolve(equation, L_initial_guess, args=CR_given) # Solve the equation to find the length
    return L_solution[0] # Return the solution for the length


if __name__ == "__main__":
    app.run() # Run the Flask application
