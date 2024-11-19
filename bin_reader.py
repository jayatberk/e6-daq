import numpy as np
from pathlib import Path

def view_binary_file(filepath, num_samples=20):
    """
    Read and display contents of a binary file containing uint64 timestamps.
    
    Args:
        filepath: Path to the binary file
        num_samples: Number of timestamps to display from start and end
    """
    try:
        # Read the binary file
        data = np.fromfile(filepath, dtype=np.uint64)
        
        # Get file size
        file_size = Path(filepath).stat().st_size
        
        print(f"\nFile Information:")
        print(f"Path: {filepath}")
        print(f"File size: {file_size:,} bytes")
        print(f"Number of timestamps: {len(data):,}")
        
        if len(data) > 0:
            print(f"\nData type: {data.dtype}")
            print(f"Min timestamp: {data.min():,}")
            print(f"Max timestamp: {data.max():,}")
            
            # Show first few timestamps
            print(f"\nFirst {num_samples} timestamps:")
            for i, value in enumerate(data[:num_samples]):
                print(f"[{i:3d}] {value:,}")
                
            if len(data) > num_samples * 2:
                print("\n...")
                
            # Show last few timestamps
            print(f"\nLast {num_samples} timestamps:")
            for i, value in enumerate(data[-num_samples:]):
                print(f"[{len(data)-num_samples+i:3d}] {value:,}")
                
            # Calculate some time differences
            time_diffs = np.diff(data[:1000])  # First 1000 differences
            print(f"\nTime difference statistics (first 1000 pairs):")
            print(f"Min difference: {time_diffs.min():,}")
            print(f"Max difference: {time_diffs.max():,}")
            print(f"Mean difference: {time_diffs.mean():,.2f}")
            
    except Exception as e:
        print(f"Error reading file: {str(e)}")

def main():
    # Replace this with your binary file path
    binary_file = r"C:\Users\jayom\Downloads\processed_photon_data\PTPhotonTimer_00005.bin"
    
    # You can adjust how many samples to show
    view_binary_file(binary_file, num_samples=10)
    
if __name__ == "__main__":
    main()