import numpy as np

# Function that accepts a list of exactly 9 numbers
# Converts this list into a 3x3 NumPy array
def calculate(input_list):
    if len(input_list) != 9:
        raise ValueError('List must contain nine numbers.')
    arr = np.array(input_list).reshape(3, 3)

    # Using NumPy functions for each statistic

    # Each metric should contain: list of calculations by columns - axis=0
    # list of calculations by rows - axis=1
    # scalar calculation for all elements - flattened

    # "mean":
    mean_col = np.mean(arr, axis=0).tolist()
    mean_row = np.mean(arr, axis=1).tolist()
    mean_flat = np.mean(arr).tolist()

    # "variance":
    var_col = np.var(arr, axis=0).tolist()
    var_row = np.var(arr, axis=1).tolist()
    var_flat = np.var(arr).tolist()

    # "standard deviation":
    std_col = np.std(arr, axis=0).tolist()
    std_row = np.std(arr, axis=1).tolist()
    std_flat = np.std(arr).tolist()

    # "max":
    max_col = np.max(arr, axis=0).tolist()
    max_row = np.max(arr, axis=1).tolist()
    max_flat = np.max(arr).tolist()

    # "min":
    min_col = np.min(arr, axis=0).tolist()
    min_row = np.min(arr, axis=1).tolist()
    min_flat = np.min(arr).tolist()

    # "sum":
    sum_col = np.sum(arr, axis=0).tolist()
    sum_row = np.sum(arr, axis=1).tolist()
    sum_flat = np.sum(arr).tolist()

    # Return the dictionary
    calculations = {
        'mean': [mean_col, mean_row, mean_flat],
        'variance': [var_col, var_row, var_flat],
        'standard deviation': [std_col, std_row, std_flat],
        'max': [max_col, max_row, max_flat],
        'min': [min_col, min_row, min_flat],
        'sum': [sum_col, sum_row, sum_flat]
    }

    return calculations

# Get the dictionary
results = calculate([0, 1, 2, 3, 4, 5, 6, 7, 8])

# Print the dictionary with line breaks
print('{')
for key, value in results.items():
    print(f"  '{key}': {value},")
print('}')
