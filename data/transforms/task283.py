import numpy as np

def find_rectangles(grid):
    """Find all rectangular regions of 5s in the grid."""
    grid = np.array(grid)
    visited = np.zeros_like(grid, dtype=bool)
    rectangles = []
    
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 5 and not visited[i, j]:
                # Find the rectangle bounds
                # Find rightmost extent
                r, c = i, j
                while r < rows and grid[r, j] == 5:
                    r += 1
                while c < cols and grid[i, c] == 5:
                    c += 1
                # Verify it's a solid rectangle
                if np.all(grid[i:r, j:c] == 5):
                    # Mark as visited
                    visited[i:r, j:c] = True
                    rectangles.append((i, r, j, c))  # top, bottom, left, right
    return rectangles

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid, dtype=int)
    result = grid.copy()
    
    # Find rectangles
    rectangles = find_rectangles(grid)
    
    for top, bottom, left, right in rectangles:
        h, w = bottom - top, right - left
        
        # Fill with 2s first
        result[top:bottom, left:right] = 2
        
        # Top and bottom edges to 4 (except corners)
        if h > 1:
            result[top, left+1:right-1] = 4
            result[bottom-1, left+1:right-1] = 4
        
        # Left and right edges to 4 (except corners)
        if w > 1:
            result[top+1:bottom-1, left] = 4
            result[top+1:bottom-1, right-1] = 4
        
        # Corners to 1
        result[top, left] = 1
        if w > 1:
            result[top, right-1] = 1
        if h > 1:
            result[bottom-1, left] = 1
            if w > 1:
                result[bottom-1, right-1] = 1
    
    return result.tolist()