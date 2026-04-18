import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Transforms the input grid by filling rows with a specific color if the 
    first and last elements of that row are identical and non-zero.
    
    Logic:
    1. Convert the input list to a numpy array for easier manipulation.
    2. Iterate through each row.
    3. Check if the first element (index 0) and the last element (index -1) 
       are equal and not zero.
    4. If they match, fill the entire row with that value.
    5. Return the transformed grid as a list of lists.
    """
    arr = np.array(grid)
    rows, cols = arr.shape
    
    for i in range(rows):
        first_val = arr[i, 0]
        last_val = arr[i, -1]
        
        # Check if both ends are non-zero and equal
        if first_val != 0 and first_val == last_val:
            arr[i, :] = first_val
            
    return arr.tolist()