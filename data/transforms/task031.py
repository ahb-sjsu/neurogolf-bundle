def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    
    min_row = rows
    max_row = -1
    min_col = cols
    max_col = -1
    
    # Find the bounding box of non-zero elements
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:
                if i < min_row:
                    min_row = i
                if i > max_row:
                    max_row = i
                if j < min_col:
                    min_col = j
                if j > max_col:
                    max_col = j
    
    # If no non-zero elements found, return empty list (though problem implies there will be some)
    if max_row == -1:
        return []
    
    # Extract the subgrid
    result = []
    for i in range(min_row, max_row + 1):
        row = grid[i][min_col:max_col + 1]
        result.append(row)
    
    return result