import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Expands a 3x3 input grid into a 9x9 output grid using a Kronecker product-like logic.
    
    For each cell (i, j) in the input grid with value v:
    - If v is 0, the corresponding 3x3 block in the output is all zeros.
    - If v is non-zero, the corresponding 3x3 block in the output is a copy of the original input grid.
    
    This effectively replaces every non-zero pixel in the input with a scaled-down version of the entire input pattern.
    """
    input_arr = np.array(grid)
    n = input_arr.shape[0]  # Should be 3 based on examples
    
    # Initialize the output array with zeros. Size is (n*n) x (n*n) -> 9x9
    output_arr = np.zeros((n * n, n * n), dtype=int)
    
    # Iterate over each cell in the input grid
    for i in range(n):
        for j in range(n):
            val = input_arr[i, j]
            
            # Calculate the top-left coordinate of the 3x3 block in the output grid
            row_start = i * n
            col_start = j * n
            
            if val != 0:
                # If the cell is non-zero, place the entire input grid into this block
                output_arr[row_start:row_start+n, col_start:col_start+n] = input_arr
            # If val is 0, the block remains zeros (already initialized)
            
    return output_arr.tolist()