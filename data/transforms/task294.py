import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    result = arr.copy()
    
    # Find connected components of 5s
    visited = np.zeros_like(arr, dtype=bool)
    rows, cols = arr.shape
    
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] == 5 and not visited[r, c]:
                # BFS/DFS to find this rectangle of 5s
                # Find bounding box of this connected component
                stack = [(r, c)]
                visited[r, c] = True
                cells = [(r, c)]
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and arr[nr, nc] == 5:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                            cells.append((nr, nc))
                            min_r = min(min_r, nr)
                            max_r = max(max_r, nr)
                            min_c = min(min_c, nc)
                            max_c = max(max_c, nc)
                
                # Fill interior with 2s
                for ir in range(min_r + 1, max_r):
                    for ic in range(min_c + 1, max_c):
                        if arr[ir, ic] == 5:  # only fill if it's part of the rectangle
                            result[ir, ic] = 2
    
    return result.tolist()