import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats
from scipy.stats import qmc, rv_discrete
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Define your parameters
parameters = {
    # In ladybug/honeybee the orientation is defined as angle anticlockwise from North.
    # Therefore, North:0, North-West:45, West:90, Sout-West:135, South:180, South-East:225, East:270, North-East:315
    "orientation": {
        "until 1945": {"values": [0, 45, 90, 135, 180, 225, 270, 315]},
        "1945-1975": {"values": [0, 45, 90, 135, 180, 225, 270, 315]},
        "1975-1995": {"values": [0, 45, 90, 135, 180, 225, 270, 315]},
        "1995 after": {"values": [0, 45, 90, 135, 180, 225, 270, 315]},
    },
    "compactness_ratio": {
        "until 1945": {
            "bins": [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],
            "probs": [0.0, 0.0, 0.412, 0.451, 0.113, 0.017, 0.007, 0]
        },
        "1945-1975": {
            "bins": [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],
            "probs": [0.0, 0.0, 0.582, 0.374, 0.042, 0.001, 0.001, 0.0]
        },
        "1975-1995": {
            "bins": [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],
            "probs": [0.0, 0.007, 0.697, 0.268, 0.028, 0.0, 0.0, 0.0]
        },
        "1995 after": {
            "bins": [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],
            "probs": [0.0, 0.0, 0.658, 0.303, 0.032, 0.007, 0.0, 0.0]
        },
    },
    "WWR": {
        "until 1945": {"values": [31]},
        "1945-1975": {"values": [36]},
        "1975-1995": {"values": [31]},
        "1995 after": {"values": [29]},
    },
    "Rc_Gr": {
        "until 1945": {"distribution": "triangular", "limits": (0.15, 5.04), "peak": 0.77},
        "1945-1975": {"distribution": "triangular", "limits": (0.15, 5.48), "peak": 0.57},
        "1975-1995": {"distribution": "triangular", "limits": (0.52, 5.38), "peak": 1.16},
        "1995 after": {"distribution": "triangular", "limits": (1.7, 6), "peak": 2.68},
    },
    "Rc_Wl": {
        "until 1945": {"distribution": "triangular", "limits": (0.19, 2.53), "peak": 0.7},
        "1945-1975": {"distribution": "triangular", "limits": (0.19, 3.5), "peak": 0.84},
        "1975-1995": {"distribution": "triangular", "limits": (0.8, 2.71), "peak": 1.53},
        "1995 after": {"distribution": "triangular", "limits": (1.51, 7), "peak": 2.68},
    },
    "Rc_Rf": {
        "until 1945": {"distribution": "triangular", "limits": (0.22, 2.53), "peak": 1.24},
        "1945-1975": {"distribution": "triangular", "limits": (0.22, 3.78), "peak": 1.22},
        "1975-1995": {"distribution": "triangular", "limits": (0.44, 3.78), "peak": 1.5},
        "1995 after": {"distribution": "triangular", "limits": (2, 9), "peak": 2.75},
    },
    "U_Gz": {
        "until 1945": {"distribution": "triangular", "limits": (1.4, 5.1), "peak": 2.96},
        "1945-1975": {"distribution": "triangular", "limits": (1.565, 5.59), "peak": 2.73},
        "1975-1995": {"distribution": "triangular", "limits": (1.8, 5.62), "peak": 2.82},
        "1995 after": {"distribution": "triangular", "limits": (1, 3.31), "peak": 2.1},
    },
    "U_Dr": {
        "until 1945": {"distribution": "triangular", "limits": (2, 3.4), "peak": 3.36},
        "1945-1975": {"distribution": "triangular", "limits": (2, 3.4), "peak": 3.31},
        "1975-1995": {"distribution": "triangular", "limits": (2, 3.4), "peak": 3.33},
        "1995 after": {"distribution": "triangular", "limits": (1, 3.4), "peak": 3.27},
    },
    "Inf": {
        "until 1945": {"distribution": "triangular", "limits": (0.7, 3), "peak": 3},
        "1945-1975": {"distribution": "triangular", "limits": (0.7, 3), "peak": 3},
        "1975-1995": {"distribution": "triangular", "limits": (0.7, 2.5), "peak": 2},
        "1995 after": {"distribution": "triangular", "limits": (0.7, 1.5), "peak": 1},
    },
    "Vent_sys": {
        # Since scipy libraries cannot take alphanumeric number we give codes to the ventilation system.
        # 1: System A, 2: System B, 3:System C1, 4:System C3c, 5:System C3a, 6: system D1, 7: System D5b
        "until 1945": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.866, 0, 0.129, 0, 0, 0.005, 0]
        },
        "1945-1975": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.791, 0, 0.207, 0, 0, 0, 0.002]
        },
        "1975-1995": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.364, 0, 0.621, 0, 0, 0.013, 0.002]
        },
        "1995 after": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.005, 0, 0.827, 0.005, 0.00, 0.154, 0.009]
        },
    },
    "Temp_set": {
        "until 1945": {"values": [18, 19, 20, 21]},
        "1945-1975": {"values": [18, 19, 20, 21]},
        "1975-1995": {"values": [18, 19, 20, 21]},
        "1995 after": {"values": [18, 19, 20, 21]},
    }
}

# Define the probabilities for each construction year
year_probabilities = {"until 1945": 0.172, "1945-1975": 0.309, "1975-1995": 0.338, "1995 after": 0.181}


def get_params():
    """
    Returns the total number of parameters defined
    
    Returns:
    int: total number of parameters
    """
    return len(parameters)

def CR_function(L, CR_target):
    # Relationship between CR and L given fixed width
    return (3.89 * L + 61.56 + 10.8 * np.sqrt(24.01 + (L/2)**2)) / (14.58 * L) - CR_target

def find_length_for_CR(CR_target, initial_guess=1):
    # Solve for L using fsolve
    L_solution, = fsolve(CR_function, x0=initial_guess, args=(CR_target,))
    return L_solution

def sample_parameters(parameters, lhs_sample):
    """
    This function picks random values for different parameters.

    Parameters:
    - parameters: A dictionary containing the parameters and their possible values.
    - lhs_sample: This is a random number between 0 and 1 from latin hypercube sampling.

    Returns:
    - A dictionary containing the sampled parameters.
    """

    # Select a construction year based on its probability
    years = list(year_probabilities.keys())
    probs = list(year_probabilities.values())

    # np.random choice selects a random item from list of years based on their probability in the dwelling stock.
    selected_year = np.random.choice(years, p=probs)

    # Prepare a dictionary to store the sampled parameters
    sampled_params = {"Construction Year": selected_year}

    # Loop through each parameter
    for i, param in enumerate(parameters):

        # Get the details of the parameter for the selected year
        param_info = parameters[param][selected_year]

        # Get the lhs value for this parameter
        lhs_value = lhs_sample[i]

        # Depending on the type of parameter...
        if "values" in param_info:
            # If the parameter has specific values, pick one of them through sample_discrete function.
            # If the parameter has probabilities then get it or provide none.
            sampled_value = sample_discrete(param_info["values"], lhs_value, param_info.get("probs"))

        elif "bins" in param_info:
            # If the parameter is binned, pick a value based on sampled_bin function.
            sampled_value = sample_binned(param_info["bins"], param_info["probs"], lhs_value)

        elif "distribution" in param_info and param_info["distribution"] == "triangular":
            # If the parameter follows a triangular distribution, pick a value using sample_triangle function.
            sampled_value = sample_triangular(param_info["limits"], param_info["peak"], lhs_value)

        else:
            raise ValueError(
                f"Invalid parameter information for {param}. Please provide either 'values', 'bins', or 'distribution'.")

        # Store the sampled value in the dictionary
        sampled_params[param] = sampled_value

    # return the dictionary containing the sampled parameters.
    return sampled_params


def sample_discrete(values, lhs_value, probs=None):
    """
    This function picks one of the discrete values based on a random number from Latin Hypercube Sampling.

    Parameters:
    - values: A list of possible values.
    - lhs_values : A random number between 0 and 1 from Latin Hypercube Sampling.
    - probs: A list of probabilities corresponding to the values (optional)

    Returns:
    - The selected value from the list.
    """
    # If no probabilities are provided, assume a uniform distribution
    if probs is None:
        probs = [1 / len(values)] * len(values)

    # Next, we create a discrete random variable with the specified values and their associated probabilities.
    # We use the rv_discrete class from scipy.stats to do this. The 'values' parameter of rv_discrete takes a tuple
    # containing two lists: the first list is the values and the second list is their associated probabilities.
    rv = rv_discrete(values=(values, probs))

    # Now we use the ppf (Percent Point Function) method of our discrete random variable to map the lhs_value
    # (which is a uniformly distributed random number between 0 and 1) to one of our discrete values.
    selected_value = rv.ppf(lhs_value)

    # Return the selected value
    return selected_value


def sample_binned(bins, probs, lhs_value):
    """
    This function is used to pick a random value from a specific range (or 'bin').
    The bin is selected based on the probabilities provided.

    Parameters:
    - bins: This is a list of ranges. Each range is represented as a tuple.
    - probs: This is a list of probabilities. Each probability corresponds to a bin.
    - lhs_sample: This is a random number between 0 and 1 from latin hypercube sampling.

    Returns:
    - A random value from the selected bin.
    """

    # Convert the bins and probabilities to numpy arrays for easier calculations
    bins_array = np.array(bins)
    probs_array = np.array(probs)

    # We use len(bins_array) instead of bins_array directly because rv_discrete expects
    # a 1-D array-like object for xk (possible outcomes). Using len(bins_array) gives us
    # an integer that represents the number of possible outcomes, which is suitable for rv_discrete.
    bin_rv = rv_discrete(values=(range(len(bins_array)), probs_array))

    # Use the ppf (Percent Point Function) method of our discrete random variable to map the lhs_value
    # (which is a uniformly distributed random number between 0 and 1) to one of our bins
    selected_bin_index = bin_rv.ppf(lhs_value)

    # Convert selected_bin_index to integer and get the corresponding bin
    selected_bin_index = int(selected_bin_index)
    selected_bin = bins_array[selected_bin_index]

    # Get the low and high values of the selected bin
    low, high = selected_bin

    # Pick a random value from within the selected bin
    random_value = np.random.uniform(low=low, high=high)

    return random_value


def sample_triangular(limits, peak, lhs_value):
    """
    Sample a parameter from a triangular distribution.

    Parameters:
    - limits: A tuple representing the lower and upper limits of the distribution.
    - peak: The peak of the distribution.

    Returns:
    - The sampled value.
    """

    left_limit, right_limit = limits
    triangular_dist = scipy.stats.triang(loc=left_limit, scale=right_limit - left_limit,
                                         c=(peak - left_limit) / (right_limit - left_limit))

    # Use the inverse CDF (also known as ppf or percent-point function) to get a sample from the triangular distribution
    sampled_value = triangular_dist.ppf(lhs_value)

    return sampled_value


def visualize_samples_and_plot_pdf_cdf(sampled_parameters):
    """
    This function creates a histogram, PDF, and CDF for each parameter.

    Parameters:
    - sampled_parameters: A dictionary containing the sampled parameters.
    """

    # For each parameter...
    for param in sampled_parameters.keys():
        if param not in ["Sample Number", "Construction Year"]:  # Exclude these columns

            param_data = sampled_parameters[param]

            # Check if there is more than one unique value in the data
            if len(set(param_data)) > 1:
                # Calculate the min and max for this parameter across all methods
                param_min = min(sampled_parameters[param])
                param_max = max(sampled_parameters[param])

                num_bins = len(param_data)

                # Calculate the PDF and CDF
                hist, bins = np.histogram(param_data, bins=num_bins, density=True)
                pdf = hist / sum(hist)
                cdf = np.cumsum(pdf)

                # Plot PDF
                plt.figure(figsize=(20, 6))
                plt.subplot(1, 2, 1)
                sns.histplot(param_data, bins='auto', kde=True, color='blue')
                plt.title(f'PDF of {param}')
                plt.xlim([param_min, param_max])

                # Plot CDF
                plt.subplot(1, 2, 2)
                plt.plot(bins[1:], cdf, color='blue')
                plt.title(f'CDF of {param}')

                plt.show()

            else:
                print(f"Skipping {param} because it has less than two unique values.")


def generate_samples(num_samples):
    # Initialize a Latin Hypercube Sampling (LHS) engine with the number of parameters
    lhs_engine = qmc.LatinHypercube(d=len(parameters))

    # Generate LHS samples
    # lhs_samples = lhs_engine.random(n=num_samples)

    # Initialize an empty list to store all samples
    all_samples_lhs = []
    
    for _ in range(num_samples):
        valid_sample_found = False
        attempts = 0
        max_attempts = 100
        
        while not valid_sample_found and attempts < max_attempts:
            lhs_sample = lhs_engine.random(n = 1)[0]
            param_samples = sample_parameters(parameters, lhs_sample)
            
            sampled_CR = param_samples['compactness_ratio']
            calculated_length = find_length_for_CR(sampled_CR)
            
            if 5 <= calculated_length <=15:
                # if the length is valid , accept this sample 
                all_samples_lhs.append(param_samples)
                valid_sample_found = True
            else:
                # increase the attempt count and generate new sample
                attempts = attempts + 1

    # Initialize an empty dictionary to store the sampled parameters
    sampled_parameters_lhs = {}

    # For each parameter...
    for param in all_samples_lhs[0]:
        sampled_parameters_lhs[param] = [sample[param] for sample in all_samples_lhs]

    # Add sample number and construction year to the dictionary
    sampled_parameters_lhs["Sample Number"] = list(range(1, num_samples + 1))
    sampled_parameters_lhs["Construction Year"] = [sample["Construction Year"] for sample in all_samples_lhs]

    # Convert the dictionary to a DataFrame
    df_lhs = pd.DataFrame(sampled_parameters_lhs)

    # Reorder the columns of the DataFrame
    df_lhs = df_lhs[["Sample Number", "Construction Year", "orientation", "compactness_ratio",
                     "WWR", "Rc_Gr", "Rc_Wl", "Rc_Rf", "U_Gz",
                     "U_Dr", "Inf", "Vent_sys", "Temp_set"]]

    return df_lhs
