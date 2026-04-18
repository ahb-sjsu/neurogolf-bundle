import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert input list to numpy array
    arr = np.array(grid)
    
    # Find all non-zero positions
    nonzero = np.argwhere(arr != 0)
    
    if len(nonzero) == 0:
        # If no non-zero elements, return a small default (though problem implies shape exists)
        return [[0]]
    
    # Determine bounding box of the non-zero region
    r_min, c_min = nonzero.min(axis=0)
    r_max, c_max = nonzero.max(axis=0)
    
    # Extract the minimal bounding box containing all non-zero values
    cropped = arr[r_min:r_max+1, c_min:c_max+1]
    
    # The examples show that the output is the top-left quadrant of the symmetric pattern.
    # The full pattern is symmetric both horizontally and vertically.
    # Output size is roughly half (rounded up) of the cropped dimensions.
    h, w = cropped.shape
    out_h = (h + 1) // 2
    out_w = (w + 1) // 2
    
    result = cropped[:out_h, :out_w]
    
    return result.tolist()