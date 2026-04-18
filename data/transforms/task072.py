import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    # Find the divider row (all 4s)
    divider_idx = np.where(grid[:, 0] == 4)[0][0]
    
    # Get top and bottom sections (each 6 rows)
    top = grid[:divider_idx]
    bottom = grid[divider_idx + 1:]
    
    # Where they differ, output 3; where same, output 0
    result = np.where(top != bottom, 3, 0)
    
    return result.tolist()