import os
import pathlib
import sqlite3
import pandas as pd
from flask import Flask
import ghhops_server as hs
from scipy.optimize import fsolve
import numpy as np

app = Flask(__name__)
hops = hs.Hops(app)


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
    abs_sql_path = os.path.abspath(sql_path)
    sql_uri = '{}?mode=ro'.format(pathlib.Path(abs_sql_path).as_uri())

    query = 'SELECT EnergyPlusVersion FROM Simulations'
    with sqlite3.connect(sql_uri, uri=True) as con:
        cursor = con.cursor()
        r = cursor.execute(query).fetchone()
        if r:
            simulation_info = r[0]
        else:
            msg = ("Cannot find the EnergyPlusVersion in the SQL file. "
                   "Please inspect query used:\n{}".format(query))
            raise ValueError(msg)

    zone_names = zone_names.split(',')
    variables = ['Zone Ideal Loads Supply Air Total Heating Energy']
    df_all_zones = pd.DataFrame()

    for zone_name in zone_names:
        index_values = [pd.read_sql(
            "SELECT ReportVariableDataDictionaryIndex FROM ReportVariableDataDictionary WHERE KeyValue LIKE '%{}%' AND VariableName = '{}'".format(
                zone_name, variable), con=con) for variable in variables]
        index_values = [item for sublist in index_values for item in
                        sublist['ReportVariableDataDictionaryIndex'].tolist()]

        if index_values:
            df_zone_data = pd.read_sql("SELECT * FROM ReportData WHERE ReportDataDictionaryIndex IN ({})".format(
                ','.join(str(index) for index in index_values)), con=con)
            df_zone_data['Zone'] = zone_name
            df_all_zones = pd.concat([df_all_zones, df_zone_data], ignore_index=True)

    sum_of_values = df_all_zones['Value'].sum()
    heating_kwh = sum_of_values * 2.7777778e-7

    return heating_kwh


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
    abs_sql_path = os.path.abspath(sql_path)
    sql_uri = '{}?mode=ro'.format(pathlib.Path(abs_sql_path).as_uri())

    query = 'SELECT EnergyPlusVersion FROM Simulations'
    with sqlite3.connect(sql_uri, uri=True) as con:
        cursor = con.cursor()
        r = cursor.execute(query).fetchone()
        if r:
            simulation_info = r[0]
        else:
            msg = ("Cannot find the EnergyPlusVersion in the SQL file. "
                   "Please inspect query used:\n{}".format(query))
            raise ValueError(msg)

    variables = ['Zone Operative Temperature']
    df_zone_data = pd.DataFrame()

    index_values = [pd.read_sql(
        "SELECT ReportVariableDataDictionaryIndex FROM ReportVariableDataDictionary WHERE KeyValue LIKE '%{}%' AND VariableName = '{}'".format(
            temp_zone_name, variable), con=con) for variable in variables]

    index_values = [item for sublist in index_values for item in sublist['ReportVariableDataDictionaryIndex'].tolist()]

    if index_values:
        df_zone_data = pd.read_sql("SELECT * FROM ReportData WHERE ReportDataDictionaryIndex IN ({})".format(
            ','.join(str(index) for index in index_values)), con=con)
        df_zone_data['Zone'] = temp_zone_name

    temperature_data = df_zone_data['Value'].to_json(orient='records')

    return temperature_data


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
    def equation(L, CR):
        return ((3.89 * L) + 61.56 + (10.8 * np.sqrt(24.01 + (L / 2) ** 2))) / (14.58 * L) - CR

    L_initial_guess = 1  # a random guess
    # noinspection PyTypeChecker
    L_solution = fsolve(equation, L_initial_guess, args=CR_given)
    return L_solution[0]


if __name__ == "__main__":
    app.run()
