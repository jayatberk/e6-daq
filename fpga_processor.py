import numpy as np
from pathlib import Path
import logging

class OpalKellyProcessor:
    def __init__(self, base_dir: str = './processed_fpga_data'):
        """Initialize the Opal Kelly processor"""
        self.processed_dir = Path(base_dir)
        self.processed_dir.mkdir(exist_ok=True)
           
    def read_fpga_data(self, filepath: Path) -> np.ndarray:
        """Read FPGA or RedPitaya data from a file, handling various formats."""
        try:
            data = np.fromfile(filepath, dtype=np.uint16)
            return data
        except Exception as e:
            logging.error(f"Error reading FPGA data from {filepath.name} as binary: {str(e)}")
            try:
                data = np.genfromtxt(filepath, delimiter=',', skip_header=0)
                return data
            except Exception as e:
                logging.error(f"Error reading FPGA data from {filepath.name} with comma delimiter: {str(e)}")
                try:
                    data = np.genfromtxt(filepath, delimiter=None, skip_header=0)
                    return data
                except Exception as e:
                    logging.error(f"Error reading FPGA data from {filepath.name} with whitespace delimiter: {str(e)}")
                    return None

    def process_file(self, filepath: Path):
        """Process a single FPGA (or RedPitaya) data file"""
        try:
            data = self.read_fpga_data(filepath)
            if data is None or len(data) == 0:
                logging.warning(f"No data found in {filepath.name}")
                return None

            if data.ndim == 1:
                processed_data = self.moving_average(data, window_size=5)
                output_filename = self.processed_dir / f"processed_{filepath.stem}.npz"
                np.savez(output_filename, data=processed_data)
                logging.info(f"Processed {filepath.name} as 1D data")
                return output_filename
            elif data.ndim == 2 and data.shape[1] == 2:
                timestamps = data[:, 0]
                values = data[:, 1]

                processed_values = self.moving_average(values, window_size=5)

                output_filename = self.processed_dir / f"processed_{filepath.stem}.npz"
                np.savez(output_filename, timestamps=timestamps, values=processed_values)
                logging.info(f"Processed {filepath.name} as 2D RedPitaya data (timestamps, values)")
                return output_filename
            else:
                # Unexpected shape
                logging.warning(f"Data in {filepath.name} has unexpected shape {data.shape}. Cannot process.")
                return None
        except Exception as e:
            logging.error(f"Error processing {filepath.name}: {str(e)}")
            return None

    def moving_average(self, data, window_size):
        """Compute the moving average of the data"""
        data = data.flatten()
        if len(data) < window_size:
            return data 
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def process_fpga_file(filepath: Path):
    """Process FPGA or RedPitaya .txt files using OpalKellyProcessor"""
    processor = OpalKellyProcessor()
    output_file = processor.process_file(filepath)
    return output_file
