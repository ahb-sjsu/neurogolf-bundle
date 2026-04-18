import numpy as np

def find_period(grid):
    """Find the period of the repeating pattern using autocorrelation."""
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find first non-zero value and its position
    mask = grid != 0
    
    # Try to find period by checking when pattern repeats
    for ph in range(1, h//2 + 1):
        for pw in range(1, w//2 + 1):
            # Check if pattern with period (ph, pw) is consistent with non-zero elements
            valid = True
            tile = {}
            for i in range(h):
                for j in range(w):
                    if grid[i,j] != 0:
                        ti, tj = i % ph, j % pw
                        if (ti, tj) in tile:
                            if tile[(ti, tj)] != grid[i,j]:
                                valid = False
                                break
                        else:
                            tile[(ti, tj)] = grid[i,j]
                if not valid:
                    break
            
            if valid and len(tile) == ph * pw:
                return ph, pw, tile
    
    return None, None, None

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    h, w = grid.shape
    
    ph, pw, tile = find_period(grid)
    
    if ph is None:
        return grid.tolist()
    
    # Reconstruct the full pattern
    result = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            result[i, j] = tile[(i % ph, j % pw)]
    
    return result.tolist()