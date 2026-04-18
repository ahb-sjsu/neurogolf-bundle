def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find all vertical lines of 5s
    lines = []  # (length, start_row, col)
    
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r][c] == 5:
                # Find the continuous line of 5s
                start = r
                while r < rows and grid[r][c] == 5:
                    r += 1
                length = r - start
                lines.append((length, start, c))
            else:
                r += 1
    
    # Sort by length descending, then by column ascending for ties
    lines.sort(key=lambda x: (-x[0], x[2]))
    
    # Assign colors 1,2,3,4 based on length ranking
    colors = [1, 2, 3, 4]
    
    # Create output grid as copy of input with 5s replaced by 0s first
    output = [[0 if grid[r][c] == 5 else grid[r][c] for c in range(cols)] for r in range(rows)]
    
    # Fill in the colored lines
    for i, (length, start, c) in enumerate(lines):
        if i < 4:
            color = colors[i]
            for r in range(start, start + length):
                output[r][c] = color
    
    return output