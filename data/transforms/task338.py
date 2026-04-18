import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    result = np.zeros_like(grid)
    n, m = grid.shape
    
    # Find connected components of 2s
    visited = np.zeros_like(grid, dtype=bool)
    
    def find_rect(start_r, start_c):
        # Find the bounding box of this rectangle
        # Look for continuous 2s forming a rectangle border
        min_r, max_r = start_r, start_r
        min_c, max_c = start_c, start_c
        
        # Expand to find full extent
        # Find rightmost 2 in same row
        c = start_c
        while c + 1 < m and grid[start_r, c + 1] == 2:
            c += 1
        max_c = c
        
        # Find bottom row with 2s at edges
        r = start_r
        while r + 1 < n and grid[r + 1, start_c] == 2 and grid[r + 1, max_c] == 2:
            r += 1
            # Check if entire row segment is 2 or if it's the border
        max_r = r
        
        # Verify it's actually a rectangle border
        # Check top and bottom edges
        for c in range(start_c, max_c + 1):
            if grid[start_r, c] != 2 or grid[max_r, c] != 2:
                return None
        
        # Check left and right edges
        for r in range(start_r, max_r + 1):
            if grid[r, start_c] != 2 or grid[r, max_c] != 2:
                return None
        
        # Check interior is 0 (or check that it's a hollow rectangle)
        # Actually, check that border is 2 and interior is 0
        for r in range(start_r + 1, max_r):
            for c in range(start_c + 1, max_c):
                if grid[r, c] != 0:
                    # Not a hollow rectangle, might be filled or something else
                    pass
        
        return (start_r, max_r, start_c, max_c)
    
    for i in range(n):
        for j in range(m):
            if grid[i, j] == 2 and not visited[i, j]:
                # Try to find rectangle starting here
                rect = find_rect(i, j)
                if rect:
                    min_r, max_r, min_c, max_c = rect
                    # Mark as visited
                    for r in range(min_r, max_r + 1):
                        for c in range(min_c, max_c + 1):
                            visited[r, c] = True
                    
                    # Fill interior with 3s
                    for r in range(min_r + 1, max_r):
                        for c in range(min_c + 1, max_c):
                            result[r, c] = 3
    
    return result.tolist()