import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    output = grid.copy()
    
    rows, cols = grid.shape
    
    # Look for 3x3 hollow square patterns: 1s on border, 0 in center
    for r in range(rows - 2):
        for c in range(cols - 2):
            # Check if this is a hollow square: all 1s on border, 0 in center
            top = grid[r, c:c+3]
            mid_left = grid[r+1, c]
            mid_center = grid[r+1, c+1]
            mid_right = grid[r+1, c+2]
            bottom = grid[r+2, c:c+3]
            
            # Check hollow square pattern: 1,1,1 / 1,0,1 / 1,1,1
            if (np.all(top == 1) and 
                mid_left == 1 and mid_center == 0 and mid_right == 1 and
                np.all(bottom == 1)):
                # Replace with plus pattern using 2s: 0,2,0 / 2,2,2 / 0,2,0
                output[r, c:c+3] = [0, 2, 0]
                output[r+1, c:c+3] = [2, 2, 2]
                output[r+2, c:c+3] = [0, 2, 0]
    
    return output.tolist()