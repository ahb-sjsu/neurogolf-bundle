import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    
    # Find the divider color (the color that forms complete lines)
    # Check rows that are all same color
    row_colors = []
    for i in range(arr.shape[0]):
        if len(set(arr[i, :])) == 1:
            row_colors.append((i, arr[i, 0]))
    
    # Check columns that are all same color  
    col_colors = []
    for j in range(arr.shape[1]):
        if len(set(arr[:, j])) == 1:
            col_colors.append((j, arr[0, j]))
    
    # The divider color appears in both full rows and full columns
    divider_color = None
    if row_colors and col_colors:
        row_divider_colors = set(c for _, c in row_colors)
        col_divider_colors = set(c for _, c in col_colors)
        common = row_divider_colors & col_divider_colors
        if common:
            divider_color = common.pop()
    
    # Find background color (most frequent non-divider color)
    unique, counts = np.unique(arr, return_counts=True)
    non_divider = [(u, c) for u, c in zip(unique, counts) if u != divider_color]
    background_color = max(non_divider, key=lambda x: x[1])[0]
    
    # Count cells by finding divider positions
    # Horizontal dividers (full rows of divider color)
    h_dividers = [i for i, c in row_colors if c == divider_color]
    # Vertical dividers (full columns of divider color)
    v_dividers = [j for j, c in col_colors if c == divider_color]
    
    # Number of cell rows = number of horizontal dividers + 1
    # But need to check if dividers are at edges
    rows = len(h_dividers) + 1
    if h_dividers and h_dividers[0] == 0:
        rows -= 1
    if h_dividers and h_dividers[-1] == arr.shape[0] - 1:
        rows -= 1
        
    cols = len(v_dividers) + 1
    if v_dividers and v_dividers[0] == 0:
        cols -= 1
    if v_dividers and v_dividers[-1] == arr.shape[1] - 1:
        cols -= 1
    
    # Actually simpler: count regions between dividers
    # Find gaps between consecutive dividers (including edges)
    h_positions = [-1] + h_dividers + [arr.shape[0]]
    v_positions = [-1] + v_dividers + [arr.shape[1]]
    
    cell_rows = sum(1 for i in range(len(h_positions)-1) if h_positions[i+1] - h_positions[i] > 1)
    cell_cols = sum(1 for j in range(len(v_positions)-1) if v_positions[j+1] - v_positions[j] > 1)
    
    # Create output
    output = np.full((cell_rows, cell_cols), background_color, dtype=int)
    
    return output.tolist()