import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    output = np.zeros_like(arr)
    
    # Check which rows are uniform (all same value)
    for i in range(arr.shape[0]):
        if np.all(arr[i] == arr[i, 0]):
            output[i] = 5
            
    return output.tolist()