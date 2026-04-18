import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    result = grid.copy()
    
    # Find the period by looking at non-zero values
    non_zero_mask = grid != 0
    if not non_zero_mask.any():
        return result.tolist()
    
    max_val = grid[non_zero_mask].max()
    period = int(max_val)
    
    # For each position, determine what value it should have
    # by looking at other cells with the same (r % period, c % period)
    rows, cols = grid.shape
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0:
                # Find the expected value by looking at pattern position
                r_mod = r % period
                c_mod = c % period
                
                # Look for a non-zero cell with same (r_mod, c_mod) pattern position
                # Check cells at same relative position in other pattern blocks
                found = False
                for dr in range(0, rows, period):
                    for dc in range(0, cols, period):
                        nr, nc = dr + r_mod, dc + c_mod
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0:
                            result[r, c] = grid[nr, nc]
                            found = True
                            break
                    if found:
                        break
    
    return result.tolist()