import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert input list to numpy array
    arr = np.array(grid, dtype=int)
    
    # Find coordinates of non-zero elements
    nonzero_coords = np.argwhere(arr != 0)
    
    if len(nonzero_coords) == 0:
        # If no non-zero elements, return a 3x6 zero grid (based on pattern, though unlikely)
        return [[0]*6 for _ in range(3)]
    
    # Get bounding box of the non-zero region
    min_row, min_col = nonzero_coords.min(axis=0)
    max_row, max_col = nonzero_coords.max(axis=0)
    
    # Extract the minimal bounding box containing the shape
    shape_crop = arr[min_row:max_row+1, min_col:max_col+1]
    
    # Create the output grid by concatenating the shape with itself horizontally
    # The output height matches the shape's height, and width is doubled
    combined = np.hstack([shape_crop, shape_crop])
    
    # Based on examples, the output is always 3 rows. 
    # However, looking closely at the examples:
    # Ex1: shape is 3x3 -> output 3x6 (3 rows, 3*2 cols)
    # Ex2: shape is 3x3 -> output 3x6
    # Ex3: shape is 3x3 -> output 3x6
    # It seems the extracted shape is always 3x3 in these examples.
    # If the problem guarantees 3x3 shapes, we just return the combined.
    # If the shape can vary but output must be 3x6, we might need padding/cropping.
    # Given the examples, the pattern is: extract shape, duplicate horizontally.
    
    return combined.tolist()