import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Transforms the input grid by adding specific patterns around non-zero pixels.
    
    Rules observed from examples:
    - If a pixel is 1: Place 7s in its 4 orthogonal neighbors (up, down, left, right).
    - If a pixel is 2: Place 4s in its 4 diagonal neighbors.
    - Other non-zero pixels (like 6, 8) remain unchanged and do not trigger patterns.
    - Patterns are only placed on 0 cells; existing non-zero cells are not overwritten.
    - Patterns are clipped to grid boundaries.
    """
    arr = np.array(grid, dtype=int)
    h, w = arr.shape
    result = arr.copy()
    
    # Directions for orthogonal neighbors (for 1s)
    ortho_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Directions for diagonal neighbors (for 2s)
    diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Find positions of 1s and 2s
    ones = np.argwhere(arr == 1)
    twos = np.argwhere(arr == 2)
    
    # Apply pattern for 1s: place 7 in orthogonal neighbors if cell is 0
    for r, c in ones:
        for dr, dc in ortho_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if result[nr, nc] == 0:
                    result[nr, nc] = 7
    
    # Apply pattern for 2s: place 4 in diagonal neighbors if cell is 0
    for r, c in twos:
        for dr, dc in diag_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if result[nr, nc] == 0:
                    result[nr, nc] = 4
    
    return result.tolist()