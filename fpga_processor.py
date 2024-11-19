import numpy as np
from pathlib import Path
import logging

class OpalKellyProcessor:
    def __init__(self, base_dir: str = './processed_fpga_data'):
        """Initialize the Opal Kelly processor"""
        self.processed_dir = Path(base_dir)
        self.processed_dir.mkdir(exist_ok=True)
           
    def read_fpga_data(self, filepath: Path) -> np.ndarray:
        """Read FPGA data from a file, handling various formats."""
        try:
            # Attempt to read as binary data (e.g., 16-bit unsigned integers)
            data = np.fromfile(filepath, dtype=np.uint16)
            return data
        except Exception as e:
            logging.error(f"Error reading FPGA data from {filepath.name} as binary: {str(e)}")
            try:
                # Attempt to read as text with comma delimiter
                data = np.genfromtxt(filepath, delimiter=',', skip_header=0)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                return data
            except Exception as e:
                logging.error(f"Error reading FPGA data from {filepath.name} with comma delimiter: {str(e)}")
                try:
                    # Attempt to read as text with whitespace delimiter
                    data = np.genfromtxt(filepath, delimiter=None, skip_header=0)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    return data
                except Exception as e:
                    logging.error(f"Error reading FPGA data from {filepath.name} with whitespace delimiter: {str(e)}")
                    return None

    def process_file(self, filepath: Path):
        """Process a single FPGA data file"""
        try:
            data = self.read_fpga_data(filepath)
            if data is None or len(data) == 0:
                logging.warning(f"No data found in {filepath.name}")
                return None

            # Example processing: Apply a simple moving average filter
            processed_data = self.moving_average(data, window_size=5)

            # Save the processed data
            output_filename = self.processed_dir / f"processed_{filepath.stem}.npz"
            np.savez(output_filename, data=processed_data)
            logging.info(f"Processed {filepath.name}")
            return output_filename  # Return the path to the processed file
        except Exception as e:
            logging.error(f"Error processing {filepath.name}: {str(e)}")
            return None

    def moving_average(self, data, window_size):
        """Compute the moving average of the data"""
        data = data.flatten()  # Ensure data is one-dimensional
        if len(data) < window_size:
            return data  # Not enough data to apply the filter
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def process_fpga_file(filepath: Path):
    """Process FPGA data using OpalKellyProcessor"""
    processor = OpalKellyProcessor()
    output_file = processor.process_file(filepath)
    return output_file
