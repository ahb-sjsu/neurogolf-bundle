import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert input list of lists to a numpy array
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Identify rows where the first and last elements are non-zero
    # We assume background is 0. The pattern triggers if both ends are non-zero.
    # Create a mask for rows that need transformation
    mask = (arr[:, 0] != 0) & (arr[:, -1] != 0)
    
    if not np.any(mask):
        return grid
    
    # Calculate center index
    center = cols // 2
    
    # Get the indices of the rows to transform
    row_indices = np.where(mask)[0]
    
    for r in row_indices:
        left_val = arr[r, 0]
        right_val = arr[r, -1]
        
        # Fill left part: from 0 to center-1
        arr[r, :center] = left_val
        
        # Fill center
        arr[r, center] = 5
        
        # Fill right part: from center+1 to end
        arr[r, center+1:] = right_val
    
    # Convert back to list of lists
    return arr.tolist()