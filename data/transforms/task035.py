import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid, copy=True)
    rows, cols = grid.shape
    
    # Find the 8-block boundaries
    eights = np.where(grid == 8)
    if len(eights[0]) == 0:
        return grid.tolist()
    
    min_row, max_row = eights[0].min(), eights[0].max()
    min_col, max_col = eights[1].min(), eights[1].max()
    
    # Find all non-zero, non-8 markers with their positions
    markers = []
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            if val != 0 and val != 8:
                markers.append((r, c, val))
    
    # For each marker, project toward the 8-block
    for mr, mc, val in markers:
        # Check if in same column as 8-block and above it
        if min_col <= mc <= max_col and mr < min_row:
            # Project down to top of 8-block
            grid[min_row, mc] = val
        # Check if in same column as 8-block and below it
        elif min_col <= mc <= max_col and mr > max_row:
            # Project up to bottom of 8-block
            grid[max_row, mc] = val
        # Check if in same row as 8-block and to the left
        elif min_row <= mr <= max_row and mc < min_col:
            # Project right to left side of 8-block
            grid[mr, min_col] = val
        # Check if in same row as 8-block and to the right
        elif min_row <= mr <= max_row and mc > max_col:
            # Project left to right side of 8-block
            grid[mr, max_col] = val
    
    return grid.tolist()