import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    result = grid.copy()
    
    # Find all connected components of 5s
    # First, find all positions with 5
    mask = (grid == 5)
    
    # Use flood fill to find connected components
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    
    rows, cols = grid.shape
    
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] and not visited[i, j]:
                # BFS to find connected component
                component = []
                stack = [(i, j)]
                visited[i, j] = True
                
                while stack:
                    ci, cj = stack.pop()
                    component.append((ci, cj))
                    
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < rows and 0 <= nj < cols and mask[ni, nj] and not visited[ni, nj]:
                            visited[ni, nj] = True
                            stack.append((ni, nj))
                
                components.append(component)
    
    # Calculate lengths and sort
    component_info = [(len(comp), comp) for comp in components]
    # Sort by length descending
    component_info.sort(key=lambda x: -x[0])
    
    # Assign values: 1 for longest, 4 for middle, 2 for shortest
    values = [1, 4, 2]
    
    for idx, (length, comp) in enumerate(component_info):
        value = values[idx]
        for (i, j) in comp:
            result[i, j] = value
    
    return result.tolist()