import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    output = np.zeros_like(grid)
    
    # Find all cross centers (value that has 4 neighbors in + shape with same value)
    rows, cols = grid.shape
    
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            center_val = grid[r, c]
            if center_val == 0:
                continue
            # Check if it's a cross center
            up = grid[r-1, c]
            down = grid[r+1, c]
            left = grid[r, c-1]
            right = grid[r, c+1]
            
            # Check if all four neighbors exist and are equal (and non-zero, different from center)
            if up != 0 and up == down == left == right and up != center_val:
                arm_val = up
                # Found a cross, expand it
                # The pattern extends 2 in each direction
                
                # Place arms (arm_val) - extending 2 cells
                for dist in [1, 2]:
                    for dr, dc in [(-dist, 0), (dist, 0), (0, -dist), (0, dist)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            output[nr, nc] = arm_val
                
                # Place center value on diagonals at distance 1 and 2
                for dist in [1, 2]:
                    for dr, dc in [(-dist, -dist), (-dist, dist), (dist, -dist), (dist, dist)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            output[nr, nc] = center_val
                
                # Place center
                output[r, c] = center_val
    
    return output.tolist()