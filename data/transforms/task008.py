import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    output = np.zeros_like(grid)
    
    # Find positions of color 8 (the 2x2 block)
    mask_8 = grid == 8
    rows_8, cols_8 = np.where(mask_8)
    
    # Find positions of color 2 (the shape to move)
    mask_2 = grid == 2
    rows_2, cols_2 = np.where(mask_2)
    
    if len(rows_8) == 0 or len(rows_2) == 0:
        return grid.tolist()
    
    # Copy 8s to output (they stay fixed)
    output[mask_8] = 8
    
    # Determine bounds of 8-block and 2-shape
    min_row_8, max_row_8 = rows_8.min(), rows_8.max()
    min_col_8, max_col_8 = cols_8.min(), cols_8.max()
    min_row_2, max_row_2 = rows_2.min(), rows_2.max()
    min_col_2, max_col_2 = cols_2.min(), cols_2.max()
    
    # Determine direction to move: find which gap is larger
    # Check vertical vs horizontal distance
    vertical_gap = min(abs(max_row_2 - min_row_8), abs(min_row_2 - max_row_8))
    horizontal_gap = min(abs(max_col_2 - min_col_8), abs(min_col_2 - max_col_8))
    
    # Actually, determine based on relative positions which axis to move along
    # Move to become adjacent
    
    # Check if 8 is above or below 2
    if max_row_8 < min_row_2:  # 8 is above 2, move 2 up
        new_min_row = max_row_8 + 1  # place directly below 8
        row_shift = new_min_row - min_row_2
        col_shift = 0
    elif min_row_8 > max_row_2:  # 8 is below 2, move 2 down
        new_max_row = min_row_8 - 1  # place directly above 8
        row_shift = new_max_row - max_row_2
        col_shift = 0
    elif max_col_8 < min_col_2:  # 8 is left of 2, move 2 left
        new_min_col = max_col_8 + 1
        row_shift = 0
        col_shift = new_min_col - min_col_2
    else:  # 8 is right of 2, move 2 right
        new_max_col = min_col_8 - 1
        row_shift = 0
        col_shift = new_max_col - max_col_2
    
    # Apply shift to 2-shape
    for r, c in zip(rows_2, cols_2):
        new_r, new_c = r + row_shift, c + col_shift
        output[new_r, new_c] = 2
    
    return output.tolist()