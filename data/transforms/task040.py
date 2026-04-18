def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    
    # Create a copy of the grid
    result = [row[:] for row in grid]
    
    # Find border colors
    top_color = grid[0][0]
    bottom_color = grid[rows-1][0]
    left_color = grid[0][0]
    right_color = grid[0][cols-1]
    
    # Check if borders are uniform
    top_uniform = all(grid[0][c] == top_color for c in range(cols))
    bottom_uniform = all(grid[rows-1][c] == bottom_color for c in range(cols))
    left_uniform = all(grid[r][0] == left_color for r in range(rows))
    right_uniform = all(grid[r][cols-1] == right_color for r in range(rows))
    
    # Process each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                # Calculate distances to each border
                dist_top = r
                dist_bottom = rows - 1 - r
                dist_left = c
                dist_right = cols - 1 - c
                
                # Find minimum distance and corresponding color
                min_dist = float('inf')
                new_color = None
                
                if top_uniform and dist_top < min_dist:
                    min_dist = dist_top
                    new_color = top_color
                if bottom_uniform and dist_bottom < min_dist:
                    min_dist = dist_bottom
                    new_color = bottom_color
                if left_uniform and dist_left < min_dist:
                    min_dist = dist_left
                    new_color = left_color
                if right_uniform and dist_right < min_dist:
                    min_dist = dist_right
                    new_color = right_color
                
                if new_color is not None:
                    result[r][c] = new_color
    
    return result