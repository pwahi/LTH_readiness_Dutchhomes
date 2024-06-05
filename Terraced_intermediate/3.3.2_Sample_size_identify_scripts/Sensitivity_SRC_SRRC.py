import pandas as pd
import numpy as np
import openturns as ot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Function for linear regression and calculate Squared
def calculate_R2(inputs, outputs):
    """
    This function calculates the R-squared value for a given set of inputs and outputs using linear regression.
    """
    model = LinearRegression().fit(inputs, outputs)
    R2 = model.score(inputs, outputs)
    return R2


# Function to calculate p-values
def calculate_p_values(input_sample, output_sample):
    """
    This function calculates the p-values for a given input and output sample using linear regression.
    """
    p_values = []
    for i in range(input_sample.getDimension()):
        sample1 = input_sample.getMarginal(i)
        sample2 = output_sample.getMarginal(0)
        test_result_collection = ot.LinearModelTest.FullRegression(sample1, sample2)
        for j in range(len(test_result_collection) - 1):  # exclude last TestResult object
            p_values.append(test_result_collection[j].getPValue())
    return p_values


# Function to perform bootstrap analysis

def bootstrap_analysis(input_sample, output_sample, analysis_type, input_cols, bootstrap_size=1000, plot=False):
    """
    This function performs a bootstrap analysis on the given input and output samples.
    It generates bootstrap samples, computes the indices for each sample based on the analysis type (SRC or SRRC),
    and calculates the lower and upper quantiles for each index.
    input_sample, output_sample : From sensitivity analysis function
    analysis_type = "SRC" or "SRRC"
    input_cols : From sensitivity analysis function
    bootstrap_size = minimum 1000 fixed, 10,000 recommended.
    plot = False : If true it also plots the confidence intervals and
    calls the plot_analysis function to plot the sensitivity analysis graph with confidence intervals.

    how to call this :
    variable_name = bootstrap_analysis(input_df, output_df, 'SRC/SRRC', input_list, 1000, True)
    variable_name --> name of the variable to store the confidence interval
    input_df --> df from sensitivity analysis function (see docstring of sensitivity_analysis function )
    output_df --> df from sensitivity analysis function(see docstring of sensitivity_analysis function )
    'SRC/SRRC' --> chosen method
    input_list --> list from sensitivity analysis function(see docstring of sensitivity_analysis function )
    1000 --> number of bootstrap samples (min 1000, recommended 10K)
    True --> For plotting graph, False for not plotting.

    """
    # Check if bootstrap_size is less than 1000
    if bootstrap_size < 1000:
        print("Bootstrap size should be at least 1000. Setting it to 1000.")
        bootstrap_size = 1000

    # Initialize an array to store the bootstrap indices
    indices_boot = ot.Sample(bootstrap_size, input_sample.getDimension())

    # Perform the bootstrap analysis
    for i in range(bootstrap_size):
        # Generate a bootstrap sample
        selection = ot.BootstrapExperiment.GenerateSelection(input_sample.getSize(), input_sample.getSize())
        X_boot = input_sample[selection]
        Y_boot = output_sample[selection]

        # Compute the indices for the bootstrap sample based on the analysis type
        corr_analysis_bootstrap = ot.CorrelationAnalysis(X_boot, Y_boot)
        if analysis_type == 'SRC':
            indices_boot[i, :] = corr_analysis_bootstrap.computeSRC()
        elif analysis_type == 'SRRC':
            indices_boot[i, :] = corr_analysis_bootstrap.computeSRRC()
        else:
            print("Invalid analysis type. Please choose either 'SRC' or 'SRRC'.")
            return None

    # Define the significance level
    alpha = 0.05

    # Compute the lower and upper quantiles for each index
    lb = indices_boot.computeQuantilePerComponent(alpha / 2.0)
    ub = indices_boot.computeQuantilePerComponent(1.0 - alpha / 2.0)

    # Create an interval for each index
    interval = ot.Interval(lb, ub)

    if plot:
        # Call plot_analysis from here
        abs_mean_indices = np.abs(np.array(indices_boot.computeMean()))
        snstvty_df = pd.DataFrame({
            'Variable': input_cols,
            analysis_type: indices_boot.computeMean(),
            'abs_' + analysis_type: abs_mean_indices,
            'p-value': calculate_p_values(input_sample, output_sample)
        })

        # Sort the DataFrame by absolute values in descending order
        snstvty_df = snstvty_df.sort_values(by='abs_' + analysis_type, ascending=False)

        plot_analysis(snstvty_df, analysis_type, R2=None, interval=interval, lb=lb)

        plt.show()

    return interval


# Function to create a bar plot of sensitivity analysis results
def plot_analysis(snstvty_df, analysis_type, R2, interval=None, lb=None):
    """
    This function creates a bar plot of the sensitivity analysis results (SRC or SRRC).
    If an interval is provided, it adds error bars to the plot representing the confidence intervals.
    """
    # Create a bar plot using seaborn for SRC and SRRC
    fig, axs = plt.subplots(figsize=(12, 8))  # Increase the width to 15

    # Increase font size for readability and set font to Arial
    plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})

    if analysis_type == 'SRC':

        snstvty_df = snstvty_df.sort_values(by='abs_SRC', ascending=False)

        # Plot for SRC
        sns.barplot(x='abs_SRC', y='Variable', data=snstvty_df, palette='viridis', ax=axs)
        for i, v in enumerate(snstvty_df['abs_SRC']):
            axs.text(v + 0.01, i + .25, str(round(v, 2)) + ', p-value: ' + str(round(snstvty_df['p-value'].iloc[i], 5)),
                     color='black', fontweight='light')
        for i, bar in enumerate(axs.patches):
            if isinstance(snstvty_df['SRC'].iloc[i], int) or isinstance(snstvty_df['SRC'].iloc[i], float):
                if snstvty_df['SRC'].iloc[i] < 0:
                    bar.set_hatch('/')
        axs.set_xlabel('SRC')
        axs.set_ylabel('Variable')
        axs.set_title(f'Sensitivity Analysis using SRC (R^2: {R2})')

        # Add error bars if interval is provided
        if interval is not None:
            error = interval.getUpperBound() - interval.getLowerBound()
            axs.errorbar(snstvty_df['abs_SRC'], range(len(lb)), xerr=error, fmt='o')

    elif analysis_type == 'SRRC':

        snstvty_df = snstvty_df.sort_values(by='abs_SRRC', ascending=False)

        # Plot for SRRC
        sns.barplot(x='abs_SRRC', y='Variable', data=snstvty_df, palette='viridis', ax=axs)
        for i, v in enumerate(snstvty_df['abs_SRRC']):
            axs.text(v + 0.01, i + .25, str(round(v, 2)) + ', p-value: ' + str(round(snstvty_df["p-value"].iloc[i], 5)),
                     color='black', fontweight='light')
        for i, bar in enumerate(axs.patches):
            if isinstance(snstvty_df['SRRC'].iloc[i], int) or isinstance(snstvty_df['SRRC'].iloc[i], float):
                if snstvty_df['SRRC'].iloc[i] < 0:
                    bar.set_hatch('/')
        axs.set_xlabel('SRRC')
        axs.set_ylabel('Variable')
        axs.set_title(f'Sensitivity Analysis using SRRC (R^2: {R2})')

        # Add error bars if interval is provided
        if interval is not None:
            error = interval.getUpperBound() - interval.getLowerBound()
            axs.errorbar(snstvty_df['abs_SRRC'], range(len(lb)), xerr=error, fmt='o')

    plt.box(False)  # Remove the border
    plt.tight_layout()  # Increase padding between plots
    plt.show()


# Function to perform sensitivity analysis
def sensitivity_analysis(xls_file, input_cols, output_col, analysis_type, plot=False):
    """
    This function loads data from an Excel file, converts it to numpy arrays, and creates input and output samples.
    It then fits a linear regression model to calculate R-squared and adjusted R-squared values.
    Depending on the analysis type (SRC or SRRC), it computes the corresponding indices, calculates p-values,
    and creates a DataFrame for easier plotting.

    WAy to call this function :
    sens_df, input_df, input_params_list, output_df = sensitivity_analysis(file_path,input_col, output_col, 'SRC/SRRC')
    sens_df --> df to store snsitivity analysis results
    input_df --> df to store matrix of input samples
    input_param_list --> list of input parameters
    output_df --> df to store output vector correspinding to the input samples
    """

    # Load your data from Excel file
    df = pd.read_excel(xls_file)

    # Assuming the second last column is the output (Y)
    X = df[input_cols]
    Y = df[output_col]

    # Convert DataFrame to numpy array
    inputs = X.to_numpy()
    outputs = Y.to_numpy().reshape(-1, 1)  # reshape for single output variable

    # Calcualting Rsqaured
    R2 = calculate_R2(inputs, outputs)

    # Create the input and output samples
    input_sample = ot.Sample(inputs)
    output_sample = ot.Sample(outputs)

    if analysis_type == 'SRC':
        # Compute SRC indices
        corr_analysis = ot.CorrelationAnalysis(input_sample, output_sample)
        SRC_point = corr_analysis.computeSRC()

        # Convert Point to numeric values
        SRC = [SRC_point[i] for i in range(SRC_point.getDimension())]

        # Calculate p-values using openturns
        p_values = calculate_p_values(input_sample, output_sample)

        # Create a DataFrame for easier plotting
        snstvty_df = pd.DataFrame({
            'Variable': X.columns,
            'SRC': SRC,
            'p-value': p_values
        })

        # Sort the DataFrame by absolute SRC values
        snstvty_df['abs_SRC'] = np.abs(snstvty_df['SRC'])

        # Sort the DataFrame by absolute SRC values in descending order
        snstvty_df = snstvty_df.sort_values(by='abs_SRC', ascending=False)

        if plot:
            plot_analysis(snstvty_df, analysis_type, R2)

        return snstvty_df, input_sample, input_cols, output_sample, R2

    elif analysis_type == 'SRRC':
        # Compute SRRC indices
        corr_analysis = ot.CorrelationAnalysis(input_sample, output_sample)
        SRRC_point = corr_analysis.computeSRRC()

        # Convert Point to numeric values
        SRRC = [SRRC_point[i] for i in range(SRRC_point.getDimension())]

        # Calculate p-values using openturns
        p_values = calculate_p_values(input_sample, output_sample)

        # Create a DataFrame for easier plotting
        snstvty_df = pd.DataFrame({
            'Variable': X.columns,
            'SRRC': SRRC,
            'p-value': p_values
        })

        # Sort the DataFrame by absolute SRRC values
        snstvty_df['abs_SRRC'] = np.abs(snstvty_df['SRRC'])

        if plot:
            plot_analysis(snstvty_df, analysis_type, R2)

        return snstvty_df, input_sample, input_cols, output_sample, R2

    else:
        print("Invalid analysis type. Please choose either 'SRC' or 'SRRC'.")
