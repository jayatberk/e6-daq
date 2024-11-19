import numpy as np
import matplotlib.pyplot as plt

def read_npz_file(filepath):
    """Read and display contents of NPZ file"""
    # Load the NPZ file
    data = np.load(filepath)
    
    # Print available arrays
    print("\nArrays in NPZ file:")
    for key in data.files:
        arr = data[key]
        print(f"\n{key}:")
        print(f"  Shape: {arr.shape}")
        if arr.size > 0:
            print(f"  Min: {arr.min()}")
            print(f"  Max: {arr.max()}")
            print(f"  Mean: {arr.mean()}")
    
    # Plot the data based on available keys
    plt.figure(figsize=(12, 8))
    
    if 'binned_times' in data.files and 'binned_counts' in data.files:
        # Plot photon counts over time
        plt.subplot(2, 1, 1)
        plt.plot(data['binned_times'] / 1e6, data['binned_counts'], 'b-')
        plt.xlabel('Time (milliseconds)')
        plt.ylabel('Counts')
        plt.title('Photon Counts vs Time')
        plt.grid(True)
    
    if 'time_diffs' in data.files:
        # Plot histogram of time differences
        plt.subplot(2, 1, 2)
        plt.hist(data['time_diffs'], bins=50, alpha=0.7)
        plt.xlabel('Time difference (nanoseconds)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Time Differences')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    npz_file = r"path_to_your_npz_file.npz"
    read_npz_file(npz_file)

if __name__ == "__main__":
    main()
