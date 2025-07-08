import numpy as np

data = np.load('000.npz')

# List all arrays stored in the file
print("Keys:", data.files)

# Print each array
for key in data.files:
    print(f"{key}:\n{data[key]}")
    if key=='depth':
        print('****************', np.max(data[key]))