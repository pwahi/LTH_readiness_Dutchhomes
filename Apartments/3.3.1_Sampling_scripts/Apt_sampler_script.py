import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats
from scipy.stats import qmc, rv_discrete
import matplotlib.pyplot as plt


# Define your parameters
parameters = {
    # In ladybug/honeybee the orientation is defined as angle anticlockwise from North.
    # Therefore, North:0, North-West:45, West:90, Sout-West:135, South:180, South-East:225, East:270, North-East:315
    "orientation": {
        "untill 1945": {"values": [0, 45, 90, 135, 180, 225, 270, 315]},
        "1945-1975": {"values": [0, 45, 90, 135, 180, 225, 270, 315]},  
        "1975-1995": {"values": [0, 45, 90, 135, 180, 225, 270, 315]},  
        "1995 after": {"values": [0, 45, 90, 135, 180, 225, 270, 315]},  
    },
    "compactness_ratio": {
        "untill 1945": {
            "bins": [(0.4, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],
            "probs": [0.029, 0.273, 0.270, 0.323, 0.089, 0.011, 0.005, 0.0]
        },
        "1945-1975": {
            "bins": [(0.4, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],
            "probs": [0.068, 0.366, 0.214, 0.274, 0.063, 0.008, 0.007, 0.0]
        }, 
        "1975-1995": {
            "bins": [(0.4, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],
            "probs" : [0.088, 0.246, 0.309, 0.253, 0.085, 0.015, 0.004, 0.0]
        }, 
        "1995 after": {
            "bins": [(0.4, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)],
            "probs" : [0.154, 0.356, 0.175, 0.231, 0.043, 0.041, 0.0, 0.0]
        }, 
    },
    "WWR": {
        "untill 1945": {"values": [32]},
        "1945-1975": {"values": [40]},  
        "1975-1995": {"values": [33]}, 
        "1995 after": {"values": [38]},
    }, 
    "Rc_Gr": {
        "untill 1945": {"distribution": "triangular", "limits": (0.15, 3.50), "peak": 0.56},
        "1945-1975": {"distribution": "triangular", "limits": (0.15, 4.15), "peak": 0.48}, 
        "1975-1995": {"distribution": "triangular", "limits": (0.52, 3.50), "peak": 1.16},
        "1995 after": {"distribution": "triangular", "limits": (0.82, 4.59), "peak": 2},
    }, 
    "Rc_Wl": {
        "untill 1945": {"distribution": "triangular", "limits": (0.19, 3.50), "peak": 0.58},
        "1945-1975": {"distribution": "triangular", "limits": (0.19, 4.18), "peak": 0.67}, 
        "1975-1995": {"distribution": "triangular", "limits": (0.8, 3.5), "peak": 1.66},
        "1995 after": {"distribution": "triangular", "limits": (1.69, 5.69), "peak": 2.61},
    },    
    "Rc_Rf": {
        "untill 1945": {"distribution": "triangular", "limits": (0.22, 3.78), "peak": 1},
        "1945-1975": {"distribution": "triangular", "limits": (0.22, 2.0), "peak": 0.96}, 
        "1975-1995": {"distribution": "triangular", "limits": (1.3, 3.78), "peak": 1.66},
        "1995 after": {"distribution": "triangular", "limits": (2.5, 3.5), "peak": 2.67},
    },    
    "U_Gz": {
        "untill 1945": {"distribution": "triangular", "limits": (1.63, 6.2), "peak": 3.11},
        "1945-1975": {"distribution": "triangular", "limits": (1.4, 5.96), "peak": 2.87}, 
        "1975-1995": {"distribution": "triangular", "limits": (1.73, 5.4), "peak": 2.91},
        "1995 after": {"distribution": "triangular", "limits": (1, 4.1), "peak": 2.16},
    },    
    "U_Dr": {
        "untill 1945": {"distribution": "triangular", "limits": (2.29, 3.4), "peak": 3.32},
        "1945-1975": {"distribution": "triangular", "limits": (2, 3.4), "peak": 3.3}, 
        "1975-1995": {"distribution": "triangular", "limits": (2, 3.4), "peak": 3.32},
        "1995 after": {"distribution": "triangular", "limits": (2, 3.4), "peak": 3.28},
    },    
    "Inf": {
        # the numbers are the position of the apartments
        # 1: inter-inter, 2:corner-inter, 3: inter-ground, 4:inter-roof, 5:corner-ground, 6:corner-roof
        "untill 1945": {
            1: {"limits": (0.35, 1.5), "peak": 1.5},
            2: {"limits": (0.455, 1.95), "peak": 1.95},
            3: {"limits": (0.35, 1.5), "peak": 1.5},
            4: {"limits": (0.42, 1.8), "peak": 1.8},
            5: {"limits": (0.455, 1.95), "peak": 1.95},
            6: {"limits": (0.49, 2.1), "peak": 2.1}
        },
        "1945-1975": {
            1: {"limits": (0.35, 1.5), "peak": 1.5},
            2: {"limits": (0.455, 1.95), "peak": 1.95},
            3: {"limits": (0.35, 1.5), "peak": 1.5},
            4: {"limits": (0.42, 1.8), "peak": 1.8},
            5: {"limits": (0.455, 1.95), "peak": 1.95},
            6: {"limits": (0.49, 2.1), "peak": 2.1}
        },
        "1975-1995": {
            1: {"limits": (0.35, 1.25), "peak": 1},
            2: {"limits": (0.455, 1.63), "peak": 1.30},
            3: {"limits": (0.35, 1.25), "peak": 1},
            4: {"limits": (0.42, 1.50), "peak": 1.20},
            5: {"limits": (0.455, 1.63), "peak": 1.30},
            6: {"limits": (0.49, 1.75), "peak": 1.40}
        },
        "1995 after": {
            1: {"limits": (0.35, 0.75), "peak": 0.50},
            2: {"limits": (0.455, 0.98), "peak": 0.65},
            3: {"limits": (0.35, 0.75), "peak": 0.50},
            4: {"limits": (0.42, 0.90), "peak": 0.60},
            5: {"limits": (0.455, 0.98), "peak": 0.65},
            6: {"limits": (0.49, 1.05), "peak": 0.70}
        },

    },    
    "Vent_sys": {
        # Since scipy libraries cannot take alphanumeric number we give codes to the ventilation system. 
        # 1: System A, 2: System B, 3:System C1, 4:System C3c, 5:System C3a, 6: system D1, 7: System D5b 
        "untill 1945": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.758, 0, 0.227, 0, 0, 0.015, 0]
        },
        "1945-1975": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.528, 0, 0.460, 0, 0, 0.007, 0.005]
        }, 
        "1975-1995": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.206, 0, 0.781, 0, 0, 0.008, 0.005]
        }, 
        "1995 after": {
            "values": [1, 2, 3, 4, 5, 6, 7],
            "probs": [0.014, 0, 0.586, 0.00, 0.195, 0.195, 0.010]
        }, 
    },
    "Temp_set": {
        "untill 1945": {"values": [18, 19, 20, 21]},
        "1945-1975": {"values": [18, 19, 20, 21]},  
        "1975-1995": {"values": [18, 19, 20, 21]},  
        "1995 after": {"values": [18, 19, 20, 21]},
        }
}

# Define the probabilities for each construction year
year_probabilities = {"untill 1945": 0.1870, "1945-1975": 0.3004, "1975-1995": 0.2464,"1995 after": 0.2662}

def get_params():
    """
    Returns the total number of parameters defined
    
    Returns:
    int: total number of parameters
    """
    return len(parameters)

def calculate_new_cr(length, width, position):
    """
    Calculates the compactness ratio (CR) based on the given length, width, and position.
    
    The compactness ratio is calculated differently depending on the position provided.
    Positions range from 1 to 6, each altering the formula used to compute the CR.
    
    Parameters:
    - length (float): The length of the apartment.
    - width (float): The width of the apartment.
    - position (int): The position identifier, ranging from 1 to 6.
    
    Returns:
    - cr(float) : The calculated compactness ratio.
    
    Raises:
    - ValueError: If an invalid position is provided.
    """
    # Calculates CR based on position
    # Position 1: Interemdiate - Intermediate
    if position == 1:
        cr = ((2.8*width)+(2.8*width))/(width*length)
        
    # Position 2: Corner - Intermediate
    elif position == 2:
        cr = ((2.8*width)+(2.8*width)+(2.8*length))/(width*length)
        
    # Position 3: Intermediate - Ground
    elif position == 3:
        cr = ((2.8*width)+(2.8*width)+ 0.7*(width*length))/(width*length)
        
    # Position 4: Intermediate - Roof
    elif position == 4:
        cr = ((2.8* width)+(2.8* width)+(width*length))/(width*length)
        
    # Position 5: Corner - Ground
    elif position == 5:
        cr = ((2.8*width)+(2.8*width)+(2.8*length)+ 0.7*(width*length))/(width*length)
        
    # Position 6: Corner - Roof
    elif position == 6:
        cr = ((2.8*width)+(2.8*width)+(2.8*length)+ (width*length))/(width*length)
    else:
        raise ValueError("Invalid position value")
    return cr

def adjust_dimesions_to_fit_area(cr, position, 
                                 min_length = 3.7, max_length = 22.2, 
                                 min_width = 3.7, max_width = 22.2, 
                                 min_area = 15, max_area = 150):
    """
    Adjusts the dimensions (length and width) to fit within a specified area range,
    given a compactness ratio (CR) and position. If the initial calculations do not
    meet the area constraints, the function iteratively searches for dimensions
    that satisfy both the area and CR criteria within given bounds.
    
    Parameters:
    - cr (float): Compactness ratio.
    - position (int): Position identifier, affects the calculation of length.
    
    - min_length (int, optional): Minimum length to consider. Default is 3.7.
    - max_length (int, optional): Maximum length to consider. Default is 22.2.
    - min_width (int, optional): Minimum width to consider. Default is 3.7.
    - max_width (int, optional): Maximum width to consider. Default is 22.2.
    - min_area (int, optional): Minimum area to consider. Default is 15.
    - max_area (int, optional): Maximum area to consider. Default is 150.
    
    Returns:
    - Tuple (float, float, float): The adjusted length, width, and area if a solution
      is found; otherwise, (None, None, None).
    """

    # Calculate Length based on Position and CR
    if position == 1:      # Intermediate - Intermediate
        Length = 5.6 / cr
    elif position == 2:    # Corner-Intermediate
        Length = 5.6 / (cr - 0.415)
    elif position == 3:    # Intermediate - Ground
        Length = 5.6 / (cr - 0.7)
    elif position == 4:    # Intermediate - Roof
        Length = 5.6 / (cr - 1)
    elif position == 5:    # Corner - Ground
        Length = 5.6 / (cr - 1.15)
    else:                   # Corner - Roof
        Length = 5.6 / (cr - 1.415)
        
    # default width
    width = 6.74
    
    # calculate initial area with default width and calcauted length 
    area = Length * width
    
    # Check if initial dimensions fit area criteria
    if min_area <= area <= max_area:
        return True
    
    # iterative search for dimensions that fit criteria
    # 1000 new lengths to be searched between min and max length 
    for new_length in np.linspace(min_length, max_length, 1000):
        # 1000 new width to be searched between min and max width 
        for new_width in np.linspace(min_width, max_width, 1000):
            new_area = new_length * new_width
            # calculate the new cr with new length and new width
            new_cr = calculate_new_cr(new_length, new_width, position)
            # check if the new dimesions satsify the area criteria and 
            # new CR is closer to provided original cr passed to the function
            if (min_area <= new_area <= max_area) and abs(new_cr - cr) < 0.001:
                    return True
                
    # Return none if no solution is found
    return False
    
def sample_parameters(parameters, lhs_sample):
    """
    This function samples parameters based on their defintiions and a 
    Latin Hyper Cube Sampling. 
    
    Args:
    - parameters (dict): A dictionary containing the parameters and their possible values.
    - lhs_sample (numpy.ndarray): This is a random number between 0 and 1 from latin hypercube sampling. 
    
    Returns:
    - dict: A dictionary containing the sampled parameters.
    """
    # Select a construction year based on its probability
    years = list(year_probabilities.keys())
    probs = list(year_probabilities.values())
    
    # np.random choice selects a random item from list of years based on their probaility in the dwelling stock. 
    selected_year = np.random.choice(years, p=probs)
    
    # Prepare a dictionary to store the sampled parameters
    sampled_params = {"Construction Year": selected_year}

    # Sample compactness ratio
    cr_info = parameters["compactness_ratio"][selected_year]
    cr_index = list(parameters.keys()).index("compactness_ratio")
    compactness_ratio = sample_binned(cr_info["bins"], cr_info["probs"], lhs_sample[cr_index])
    
    # sample position based on bounds for compactness ratio
    # These bounds are calcaulated from the area bounds of 25-150 m2
    possible_positions = []
    if compactness_ratio >= 0.252 and compactness_ratio <= 1.514 :  # Intermediate - Intermediate
        possible_positions.append(1)
    if compactness_ratio >= 0.667 and compactness_ratio <= 1.929 :  # Corner-Intermediate
        possible_positions.append(2)
    if compactness_ratio >= 0.952 and compactness_ratio <= 2.214 :  # Intermediate-ground
        possible_positions.append(3)
    if compactness_ratio >= 1.252 and compactness_ratio <= 2.514 :  # Intermediate-Roof
        possible_positions.append(4)
    if compactness_ratio >= 1.402 and compactness_ratio <= 2.664 :  # Corner-Ground
        possible_positions.append(5)
    if compactness_ratio >= 1.667 and compactness_ratio <= 2.929 :  # Corner - Roof
        possible_positions.append(6)
    if compactness_ratio > 2.929:  # Corner - Roof
        possible_positions.append(6)
        
    # sampling from the list of possible positions as per the comapctness ratio
    position= sample_discrete(possible_positions, np.random.rand())
        
    area_check = adjust_dimesions_to_fit_area(compactness_ratio, position)
    
    if not area_check:
        return ("Area crietria not met")
        
    # If the area_check condition is satisfied, continue sampling other parameter
    sampled_params["compactness_ratio"] = compactness_ratio
    sampled_params['position'] = position
            
    # Loop through each parameter
    for i, param in enumerate(parameters): 
        # skip comapctness ratio and position as they have been sampled. 
        # skip infilteration (Inf) parameter in the initial loop
        if param in ['compactness_ratio', 'position', 'Inf']: 
            continue
        
        # Get the details of the parameter for the selected year
        param_info = parameters[param][selected_year]
        
        # Get the lhs value for this parameter
        lhs_value = lhs_sample[i]
        
        # Sampling the parameter value based on its type (discrete, binned, triangular)
        if "values" in param_info:
            # If the parameter has specific values, pick one of them through sample_discrete function.
            # If the parameter has probabilites then get it or provide none. 
            sampled_value = sample_discrete(param_info["values"], lhs_value, param_info.get("probs"))
    
        elif "bins" in param_info:
             # If the parameter is binned, pick a value based on sampled_bin function. 
            sampled_value = sample_binned(param_info["bins"], param_info["probs"], lhs_value)
            
        elif "distribution" in param_info and param_info["distribution"] == "triangular":
            # If the parameter follows a triangular distribution, pick a value using sample_triangle function.
            sampled_value = sample_triangular(param_info["limits"], param_info["peak"], lhs_value)
        
        else:
            raise ValueError(f"Invalid parameter information for {param}. Please provide either 'values', 'bins', or 'distribution'.")
       
        # Store the sampled value in the dictionary     
        sampled_params[param] = sampled_value
        
        # Sampling infilteration based on construction year and position of the apartment
        if position is not None:
            # Infilteration range based on sampled construction year and position of the apartment. 
            inf_params = parameters['Inf'][selected_year][position]
            # sampling using the triangular distribution 
            sampled_inf = sample_triangular(inf_params["limits"], inf_params["peak"], lhs_value)
            # Adding the sampled infilteration rate to the dictionary
            sampled_params["Inf"] = sampled_inf
        
    # return the dictionary conatining the sampled parameters.
    return sampled_params

def sample_discrete(values, lhs_value, probs = None):
    """
    This function picks one of the discrete values based on a random number from Latin Hypercube Sampling.
    
    Parameters:
    - values(list) : A list of discrete values.
    - lhs_values (float) : A random number between 0 and 1 from Latin Hypercube Sampling.
    - probs (list, optional): Probabilites associated with each value in 'values'
    
    Returns:
    - The selected value from 'values'.
    """
    # If no probabilities are provided, assume a uniform distribution
    if probs is None:
        probs = [1/len(values)]*len(values)
    
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
    - bins (lsit of tuples): List of bins, eac represented by a (low, high) tuple.
    - probs (list): This is a list of probabilities for each bin.
    - lhs_sample (float): This is a random number between 0 and 1 from latin hypercube sampling, 
    used to select and sample within a bin
    
    Returns:
    - A random value within the selected bin.
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
    Sample a value from a triangular distribution defined by its limits and peak.
    
    Parameters:
    - limits (tuple): The lower and upper limits of the distribution.
    - peak (float): The peak of the triangualr distribution.
    
    Returns:
    - A value sampled from the trianguale distribution.
    """
    
    left_limit, right_limit = limits
    triangular_dist = scipy.stats.triang(loc=left_limit, scale=right_limit-left_limit, c=(peak-left_limit)/(right_limit-left_limit))
    
    # Use the inverse CDF (also known as ppf or percent-point function) to get a sample from the triangular distribution
    sampled_value = triangular_dist.ppf(lhs_value)
    
    return sampled_value

def visualize_samples_and_plot_pdf_cdf(sampled_parameters):
    """
    Visualizes the distribution of sampled parameters by plotting their PDFs and CDFs.

    Parameters:
    sampled_parameters (dict): Dictionary with parameter names as keys and lists of sampled values as values.
    """
    
    # Loop through each parameter to visualise its distribution
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
    """
    Generates a specified number of samples for the parameters using Latin Hypercube Sampling (LHS).

    Parameters:
        num_samples (int): Number of samples to generate.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated samples.
    """
    # Initialize a Latin Hypercube Sampling (LHS) engine with the number of parameters
    lhs_engine = qmc.LatinHypercube(d=len(parameters))

    # Generate LHS samples
    lhs_samples = lhs_engine.random(n=num_samples)

    # Initialize an empty list to store all samples
    all_samples_lhs = []

    # For each LHS sample, generate parameter samples and add them to the list
    for lhs_sample in lhs_samples:
        param_samples = sample_parameters(parameters, lhs_sample)
        all_samples_lhs.append(param_samples)

    # Initialize an empty dictionary to store the sampled parameters
    sampled_parameters_lhs = {}

    # For each parameter, extract the samples from all_samples_lhs and store them in the dictionary
    for param in all_samples_lhs[0]:
        sampled_parameters_lhs[param] = [sample[param] for sample in all_samples_lhs]

    # Add sample number and construction year to the dictionary
    sampled_parameters_lhs["Sample Number"] = list(range(1, num_samples + 1))
    sampled_parameters_lhs["Construction Year"] = [sample["Construction Year"] for sample in all_samples_lhs]

    # Convert the dictionary to a DataFrame
    df_lhs = pd.DataFrame(sampled_parameters_lhs)

    # Reorder the columns of the DataFrame
    df_lhs = df_lhs[["Sample Number", "Construction Year", "orientation", "compactness_ratio",
                     "position", "WWR", "Rc_Gr", "Rc_Wl", "Rc_Rf", "U_Gz",
                     "U_Dr", "Inf", "Vent_sys", "Temp_set"]]
    return df_lhs