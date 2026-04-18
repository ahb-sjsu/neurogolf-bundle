import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    result = arr.copy()
    
    # Find all positions with 8
    eight_positions = np.argwhere(arr == 8)
    
    # If there are no 8s, return as is
    if len(eight_positions) == 0:
        return result.tolist()
    
    # Find bounding box of 8s
    min_row = eight_positions[:, 0].min()
    max_row = eight_positions[:, 0].max()
    min_col = eight_positions[:, 1].min()
    max_col = eight_positions[:, 1].max()
    
    # Inside the bounding box, change 1s to 3s
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if result[r, c] == 1:
                result[r, c] = 3
    
    return result.tolist()