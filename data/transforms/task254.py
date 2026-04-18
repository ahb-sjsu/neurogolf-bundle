import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    rows, cols = arr.shape
    output = np.zeros_like(arr)
    
    # Find all vertical lines of 5s
    line_info = []  # (length, start_row, col)
    
    for c in range(cols):
        # Find contiguous vertical segments of 5s
        r = 0
        while r < rows:
            if arr[r, c] == 5:
                # Start of a line
                start = r
                while r < rows and arr[r, c] == 5:
                    r += 1
                length = r - start
                line_info.append((length, start, c))
            else:
                r += 1
    
    # Find longest and shortest lines
    if len(line_info) >= 1:
        # Sort by length
        line_info.sort(key=lambda x: x[0])
        
        # Shortest gets 2
        shortest_len, shortest_start, shortest_col = line_info[0]
        output[shortest_start:shortest_start + shortest_len, shortest_col] = 2
        
        # Longest gets 1
        longest_len, longest_start, longest_col = line_info[-1]
        output[longest_start:longest_start + longest_len, longest_col] = 1
    
    return output.tolist()