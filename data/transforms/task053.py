import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    h, w = arr.shape
    
    # Create output grid filled with zeros
    output = np.zeros_like(arr)
    
    # Shift everything down by 1 row
    # Row i in input goes to row i+1 in output (for i < h-1)
    if h > 1:
        output[1:, :] = arr[:-1, :]
    
    return output.tolist()