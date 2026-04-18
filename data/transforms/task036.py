import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Identifies a distinct 3x3 (or similar small) pattern formed by a unique color
    that appears in a cluster within the large grid, while other colors are scattered noise.
    Returns the cropped subgrid containing this pattern.
    """
    arr = np.array(grid)
    
    # Count occurrences of each non-zero color
    unique, counts = np.unique(arr[arr != 0], return_counts=True)
    
    # The target color is likely the one forming a compact shape (more than 1 pixel, 
    # but not the most frequent if the most frequent is scattered noise like 1s or 2s).
    # However, looking at the examples:
    # Ex1: 3s form a 5x3 block. 1s and 5s are scattered.
    # Ex2: 4s form a 3x3 block. 2s are scattered.
    # The target color is the one that forms the largest connected component or 
    # simply the color that isn't the "background noise" color which usually has count >> 10.
    # A simpler heuristic: The target pattern is the only dense cluster.
    # Let's find the color with the smallest bounding box relative to its count, 
    # or just look for the color that forms a rectangular block > 1x1.
    
    target_color = None
    best_score = -1
    
    for color in unique:
        coords = np.argwhere(arr == color)
        if len(coords) < 2:
            continue
        
        # Calculate bounding box
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        area = height * width
        count = len(coords)
        
        # Density score: how much of the bounding box is filled with this color
        density = count / area
        
        # We want a high density and a count > 1. 
        # In examples, the noise colors (1, 2, 5) are very scattered (low density).
        # The target colors (3, 4) form solid blocks (high density).
        if density > best_score:
            best_score = density
            target_color = color
            target_bbox = (r_min, r_max, c_min, c_max)

    if target_color is None:
        return [] # Should not happen based on problem description

    r_min, r_max, c_min, c_max = target_bbox
    
    # Extract the subgrid
    subgrid = arr[r_min:r_max+1, c_min:c_max+1]
    
    # Convert back to list of lists
    return subgrid.tolist()