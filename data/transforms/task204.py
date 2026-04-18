import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    result = grid.copy()
    
    # Find all rectangular boxes made of 1s
    rows, cols = grid.shape
    
    # Find connected components of 1s that form rectangles
    # We'll find boxes by looking for patterns: closed rectangles of 1s with 0s inside
    
    visited = np.zeros_like(grid, dtype=bool)
    
    boxes = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1 and not visited[r, c]:
                # Try to find a rectangular box
                # Find the bounding box of this connected component
                # Use BFS/DFS to find all connected 1s
                stack = [(r, c)]
                component = []
                visited[r, c] = True
                
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                
                # Check if this forms a rectangular frame (hollow rectangle)
                if len(component) >= 4:
                    min_r = min(p[0] for p in component)
                    max_r = max(p[0] for p in component)
                    min_c = min(p[1] for p in component)
                    max_c = max(p[1] for p in component)
                    
                    height = max_r - min_r + 1
                    width = max_c - min_c + 1
                    
                    # Check if it's a hollow rectangle (frame)
                    expected_frame = 2*height + 2*width - 4  # perimeter
                    
                    # Verify it's actually a frame (1s on border, checking if 0s inside or not filled)
                    is_frame = True
                    interior_1s = 0
                    for rr in range(min_r+1, max_r):
                        for cc in range(min_c+1, max_c):
                            if grid[rr, cc] == 1:
                                interior_1s += 1
                    
                    # It's a frame if interior has no 1s (or we're looking at the outline)
                    if interior_1s == 0 and len(component) == expected_frame:
                        interior_height = max_r - min_r - 1
                        interior_width = max_c - min_c - 1
                        boxes.append((min_r+1, max_r, min_c+1, max_c, interior_height, interior_width))
    
    # Determine fill value: 7 for smallest area, 2 for others? 
    # Actually: 7 for odd interior dimensions, 2 for even
    # Or: check if both dimensions are odd -> 7, else -> 2
    
    for r1, r2, c1, c2, ih, iw in boxes:
        # Fill interior
        if ih > 0 and iw > 0:
            # Determine color: if both interior dimensions are odd -> 7, else -> 2
            if ih % 2 == 1 and iw % 2 == 1:
                fill_val = 7
            else:
                fill_val = 2
            result[r1:r2, c1:c2] = fill_val
    
    return result.tolist()