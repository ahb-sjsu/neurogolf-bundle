import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    output = np.copy(grid)
    rows, cols = grid.shape
    
    # Find all positions of 5
    fives = np.argwhere(grid == 5)
    
    # For each 5, place the 3x3 pattern around it
    for r, c in fives:
        # Pattern: 5 at corners, 1 at edges, 0 at center
        # Positions relative to center (r,c)
        pattern = [
            (-1, -1, 5), (-1, 0, 1), (-1, 1, 5),
            (0, -1, 1),  (0, 0, 0),  (0, 1, 1),
            (1, -1, 5),  (1, 0, 1),  (1, 1, 5)
        ]
        
        for dr, dc, val in pattern:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr, nc] = val
    
    return output.tolist()