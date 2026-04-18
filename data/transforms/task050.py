import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    output = arr.copy()
    
    # Find all positions of 8s
    eight_positions = np.argwhere(arr == 8)
    
    if len(eight_positions) < 2:
        return grid
    
    # Group 8s by row
    row_groups = {}
    for r, c in eight_positions:
        if r not in row_groups:
            row_groups[r] = []
        row_groups[r].append(c)
    
    # Group 8s by column
    col_groups = {}
    for r, c in eight_positions:
        if c not in col_groups:
            col_groups[c] = []
        col_groups[c].append(r)
    
    # For each row with exactly 2 eights, fill horizontally between them
    for r, cols in row_groups.items():
        if len(cols) == 2:
            c1, c2 = sorted(cols)
            output[r, c1+1:c2] = 3
    
    # For each column with exactly 2 eights, fill vertically between them
    for c, rows in col_groups.items():
        if len(rows) == 2:
            r1, r2 = sorted(rows)
            output[r1+1:r2, c] = 3
    
    return output.tolist()