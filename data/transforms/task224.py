import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    output = grid.copy()
    
    # Find positions of 5s
    five_positions = np.argwhere(grid == 5)
    
    # Find the color of the inner shape (non-0, non-5)
    inner_color = None
    for val in [1, 2, 3, 4, 6, 7, 8, 9]:
        if val in grid:
            positions = np.argwhere(grid == val)
            if len(positions) > 0:
                # Check it's not just 5s
                inner_color = val
                break
    
    if inner_color is None or len(five_positions) == 0:
        return output.tolist()
    
    # Get bounds for the outer rectangle
    # The 5s define the outer corners, rectangle is just inside
    five_rows = five_positions[:, 0]
    five_cols = five_positions[:, 1]
    
    # Find which 5s are on top/bottom and left/right
    top_row = min(five_rows) + 1  # rectangle starts below top 5
    bottom_row = max(five_rows) - 1  # rectangle ends above bottom 5
    
    # Need to find left and right cols from 5s that aren't in middle rows
    # Actually looking more carefully: left 5 is at same row as inner shape, right 5 too in ex1
    
    # The 5s seem to be at: top, bottom, left-edge-row, right-edge-row
    # Find leftmost and rightmost 5 positions that aren't top/bottom
    middle_fives = [(r, c) for r, c in five_positions if r > min(five_rows) and r < max(five_rows)]
    if middle_fives:
        left_col = min(c for r, c in middle_fives) - 1  # rectangle starts right of left 5? No, check ex1
        # In ex1: 5 at col 1, rectangle starts at col 2. 5 at col 11, rectangle ends at col 10.
        left_col = min(c for r, c in middle_fives) + 1
        right_col = max(c for r, c in middle_fives) - 1
    else:
        # Use all 5s to determine
        left_col = min(five_cols) + 1
        right_col = max(five_cols) - 1
    
    # Actually re-checking examples more carefully:
    # Ex1: 5s at rows 1,6,6,11 and cols 4,1,11,4
    # Rectangle rows 2-10, cols 2-10
    
    # Let me recalculate: min row with 5 is 1, max is 11. Rectangle rows 2 to 10 (exclusive of 5 rows + 1 each side)
    # For cols: 5s at 1,4,4,11. But 4s are middle rows. 
    # Actually left 5 is at col 1 (row 6), right 5 at col 11 (row 6)
    # Rectangle cols 2 to 10
    
    # So: rectangle is between the 5s, exclusive
    top_five_rows = [r for r, c in five_positions if r == min(five_rows)]
    bottom_five_rows = [r for r, c in five_positions if r == max(five_rows)]
    
    # Get all row positions and col positions of 5s
    row_5s = sorted(set(five_rows))
    col_5s = sorted(set(five_cols))
    
    # Find rectangle bounds
    rect_top = row_5s[0] + 1
    rect_bottom = row_5s[-1] - 1
    
    # For columns, need to find which 5s are on the sides vs top/bottom
    # Check: 5s that share rows with inner shape vs unique rows
    inner_positions = np.argwhere(grid == inner_color)
    inner_rows = set(inner_positions[:, 0])
    
    side_fives = [(r, c) for r, c in five_positions if r in inner_rows]
    if len(side_fives) >= 2:
        rect_left = min(c for r, c in side_fives) + 1
        rect_right = max(c for r, c in side_fives) - 1
    else:
        rect_left = col_5s[0] + 1
        rect_right = col_5s[-1] - 1
    
    # Draw the outer rectangle
    # Top and bottom edges
    output[rect_top, rect_left:rect_right+1] = inner_color
    output[rect_bottom, rect_left:rect_right+1] = inner_color
    
    # Left and right edges  
    output[rect_top:rect_bottom+1, rect_left] = inner_color
    output[rect_top:rect_bottom+1, rect_right] = inner_color
    
    return output.tolist()