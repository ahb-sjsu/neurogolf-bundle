import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    rows, cols = arr.shape
    out = np.zeros_like(arr)

    # Find all unique non-zero colors
    colors = np.unique(arr[arr != 0])

    for color in colors:
        # Mask of current color
        mask = (arr == color)
        # Get coordinates of pixels of this color
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        # For each row that contains this color, find min and max column
        unique_rows = np.unique(ys)
        for r in unique_rows:
            row_xs = xs[ys == r]
            c_min, c_max = row_xs.min(), row_xs.max()
            # Fill horizontally between min and max with the color
            out[r, c_min:c_max+1] = color

    return out.tolist()