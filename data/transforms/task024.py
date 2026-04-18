import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert input list of lists to numpy array
    grid_np = np.array(grid)
    rows, cols = grid_np.shape
    
    # Initialize output grid with zeros
    output_np = np.zeros((rows, cols), dtype=int)
    
    # Lists to store row and column fill instructions
    row_fills = [] # List of (row_idx, value)
    col_fills = [] # List of (col_idx, value)
    
    # Scan the input grid to identify triggers
    # We iterate through every cell
    for r in range(rows):
        for c in range(cols):
            val = grid_np[r, c]
            if val == 1 or val == 3:
                row_fills.append((r, val))
            elif val == 2:
                col_fills.append((c, val))
    
    # Apply column fills first (Value 2 -> Vertical)
    for c_idx, val in col_fills:
        output_np[:, c_idx] = val
        
    # Apply row fills second (Value 1, 3 -> Horizontal), overwriting columns if necessary
    for r_idx, val in row_fills:
        output_np[r_idx, :] = val
        
    # Convert back to list of lists
    return output_np.tolist()