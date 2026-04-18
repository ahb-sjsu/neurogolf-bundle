import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Transforms the input grid by filling enclosed regions of 0s surrounded by 3s with 4s.
    
    The logic identifies connected components of 0s. If a component of 0s does not 
    touch the boundary of the grid, it is considered "enclosed" by the non-zero values 
    (specifically 3s in the examples) and is filled with 4s.
    """
    arr = np.array(grid, dtype=int)
    rows, cols = arr.shape
    
    # We are looking for 0s that are completely surrounded by non-zeros (3s).
    # This is equivalent to finding background (0s connected to the edge) vs holes (0s not connected).
    
    # Create a mask of all zeros
    is_zero = (arr == 0)
    
    # If there are no zeros, return original
    if not np.any(is_zero):
        return grid
    
    # Perform a flood fill (BFS/DFS) from the boundaries to find all 0s connected to the edge.
    # These are the "background" zeros that should NOT be changed.
    visited = np.zeros((rows, cols), dtype=bool)
    stack = []
    
    # Initialize stack with all boundary cells that are 0
    for r in range(rows):
        for c in [0, cols - 1]:
            if arr[r, c] == 0:
                stack.append((r, c))
                visited[r, c] = True
                
    for c in range(cols):
        for r in [0, rows - 1]:
            if arr[r, c] == 0 and not visited[r, c]:
                stack.append((r, c))
                visited[r, c] = True
    
    # DFS to mark all connected background zeros
    while stack:
        r, c = stack.pop()
        # Check 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr, nc] and arr[nr, nc] == 0:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
    
    # The zeros that are NOT visited are the enclosed ones.
    # Change them to 4.
    enclosed_zeros = is_zero & (~visited)
    arr[enclosed_zeros] = 4
    
    return arr.tolist()