import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Transforms the input grid by projecting the pattern of 5s from the first row
    onto specific target rows, changing the projected 5s to 2s.

    Logic observed from examples:
    1. Identify the "source pattern" from the first row (row 0). This pattern consists
       of columns where the value is 5.
    2. Identify "target rows". These are rows that have a 5 in the last column (index -1).
       Note: In the examples, the first row (row 0) also has the pattern but is NOT treated
       as a target row for modification (it keeps its 5s). The transformation only happens
       in rows below row 0 that are marked by a 5 in the last column.
    3. For each identified target row:
       - Copy the source pattern's structure.
       - Where the source row has a 5, place a 2 in the target row.
       - The 5 in the last column of the target row is preserved.
       - Other cells remain 0.
    """
    # Convert to numpy array for easier manipulation
    arr = np.array(grid, dtype=int)
    rows, cols = arr.shape
    
    # Get the source pattern from the first row (row 0)
    # We create a boolean mask where the first row equals 5
    source_mask = (arr[0, :] == 5)
    
    # Identify target rows: rows that have a 5 in the last column.
    # We exclude row 0 because row 0 is the source and should remain unchanged (keeps 5s).
    # The examples show that row 0 is never converted to 2s even if it has a 5 at the end 
    # (though in examples, row 0 doesn't have a 5 at the end, but logically we only modify 
    # rows that are "receivers").
    # Actually, looking closely at Example 1:
    # Row 3 has a 5 at the end. Row 7 has a 5 at the end.
    # These rows get the pattern.
    # Row 0 has the pattern originally.
    
    target_row_indices = []
    for r in range(1, rows):  # Start from 1 to skip the source row
        if arr[r, -1] == 5:
            target_row_indices.append(r)
    
    # Apply the transformation
    for r in target_row_indices:
        # Create a new row initialized with zeros (or copy existing if needed, but examples show overwrite except last col)
        # The examples show that the existing 5 at the last column is preserved.
        # The rest of the row becomes 0s, then 2s are placed based on the mask.
        
        new_row = np.zeros(cols, dtype=int)
        
        # Place 2s where the source mask is True
        new_row[source_mask] = 2
        
        # Preserve the marker in the last column (which is 5)
        new_row[-1] = 5
        
        # Update the grid
        arr[r, :] = new_row

    return arr.tolist()