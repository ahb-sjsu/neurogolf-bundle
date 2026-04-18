import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    out = np.zeros_like(grid)
    
    # Find the non-zero position in bottom row
    start_col = np.where(grid[-1] != 0)[0][0]
    color = grid[-1, start_col]
    
    # Create vertical stripes every 2 columns starting from start_col
    for c in range(start_col, 10, 2):
        out[:, c] = color
    
    # Fill gaps with 5s at top or bottom alternating
    gap_cols = list(range(start_col + 1, 10, 2))
    for i, c in enumerate(gap_cols):
        if i % 2 == 0:  # even index gap -> top row
            out[0, c] = 5
        else:  # odd index gap -> bottom row
            out[-1, c] = 5
    
    return out.tolist()