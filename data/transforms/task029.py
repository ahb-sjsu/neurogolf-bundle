import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    
    # Find the border color and frame
    # The border color appears at least twice in same row and same column
    rows, cols = grid.shape
    
    # Find candidate border colors (non-background colors that form lines)
    for color in range(10):
        # Find rows where this color appears multiple times
        row_positions = []
        for r in range(rows):
            positions = np.where(grid[r, :] == color)[0]
            if len(positions) >= 2:
                row_positions.append((r, positions.min(), positions.max()))
        
        # Find columns where this color appears multiple times  
        col_positions = []
        for c in range(cols):
            positions = np.where(grid[:, c] == color)[0]
            if len(positions) >= 2:
                col_positions.append((c, positions.min(), positions.max()))
        
        if len(row_positions) >= 2 and len(col_positions) >= 2:
            # Check if they form a rectangle
            # Get top and bottom rows with this color spanning many columns
            row_positions_sorted = sorted(row_positions, key=lambda x: x[0])
            top_row, top_left, top_right = row_positions_sorted[0]
            bottom_row, bottom_left, bottom_right = row_positions_sorted[-1]
            
            # Get left and right columns with this color spanning many rows
            col_positions_sorted = sorted(col_positions, key=lambda x: x[0])
            left_col, left_top, left_bottom = col_positions_sorted[0]
            right_col, right_top, right_bottom = col_positions_sorted[-1]
            
            # Verify it forms a consistent rectangle
            if (top_row == left_top == right_top and 
                bottom_row == left_bottom == right_bottom and
                left_col == top_left == bottom_left and
                right_col == top_right == bottom_right):
                
                # Check that the borders are filled with this color
                top_border = grid[top_row, left_col:right_col+1]
                bottom_border = grid[bottom_row, left_col:right_col+1]
                left_border = grid[top_row:bottom_row+1, left_col]
                right_border = grid[top_row:bottom_row+1, right_col]
                
                if (np.all(top_border == color) and 
                    np.all(bottom_border == color) and
                    np.all(left_border == color) and
                    np.all(right_border == color)):
                    
                    # Extract interior
                    interior = grid[top_row+1:bottom_row, left_col+1:right_col]
                    return interior.tolist()
    
    # Fallback: try simpler approach - find any color forming a rectangular frame
    from collections import Counter
    
    for color in range(10):
        # Find all positions of this color
        positions = np.argwhere(grid == color)
        if len(positions) < 4:
            continue
            
        rows_with_color = positions[:, 0]
        cols_with_color = positions[:, 1]
        
        min_row, max_row = rows_with_color.min(), rows_with_color.max()
        min_col, max_col = cols_with_color.min(), cols_with_color.max()
        
        # Check if forms a frame
        frame_size = 2 * (max_row - min_row + 1) + 2 * (max_col - min_col + 1) - 4
        if len(positions) != frame_size:
            continue
            
        # Verify it's actually a frame
        top_edge = grid[min_row, min_col:max_col+1]
        bottom_edge = grid[max_row, min_col:max_col+1]
        left_edge = grid[min_row:max_row+1, min_col]
        right_edge = grid[min_row:max_row+1, max_col]
        
        if (np.all(top_edge == color) and 
            np.all(bottom_edge == color) and
            np.all(left_edge == color) and
            np.all(right_edge == color)):
            
            interior = grid[min_row+1:max_row, min_col+1:max_col]
            return interior.tolist()
    
    return []