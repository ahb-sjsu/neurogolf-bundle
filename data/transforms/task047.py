import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    result = arr.copy()
    
    # Find positions of 8 and 7
    pos8 = np.where(arr == 8)
    pos7 = np.where(arr == 7)
    
    row8, col8 = pos8[0][0], pos8[1][0]
    row7, col7 = pos7[0][0], pos7[1][0]
    
    # Draw horizontal line through row8 (filled with 8s)
    result[row8, :] = 8
    # Except at intersection with col7, make it 2
    result[row8, col7] = 2
    
    # Draw horizontal line through row7 (filled with 7s)
    result[row7, :] = 7
    # Except at intersection with col8, make it 2
    result[row7, col8] = 2
    
    # Draw vertical line through col8 (filled with 8s)
    result[:, col8] = 8
    # Fix intersection points that should be 2
    result[row8, col8] = 8  # This is the original 8 position
    result[row7, col8] = 2  # Intersection with row7
    
    # Draw vertical line through col7 (filled with 7s)
    result[:, col7] = 7
    # Fix intersection points that should be 2
    result[row7, col7] = 7  # This is the original 7 position
    result[row8, col7] = 2  # Intersection with row8
    
    return result.tolist()