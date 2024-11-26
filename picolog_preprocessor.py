import numpy as np
from pathlib import Path
from scipy import signal
import logging
import uuid  # Added to generate unique filenames

class PicologPreprocessor:
    def __init__(self, processed_directory):
        self.processed_directory = Path(processed_directory)
        self.processed_directory.mkdir(exist_ok=True)

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        """Apply Butterworth low-pass filter to data"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        if normal_cutoff >= 1:
            raise ValueError("Filter critical frequency is too high.")
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data)

    def process_file(self, filepath):
        """Process a single Photon Timer binary file"""
        try:
            # Read the binary data
            data = self.read_binary_file(filepath)

            if data is None or len(data) == 0:
                logging.warning(f"No data found in {filepath.name}")
                return None

            # Assuming the data contains timestamps in picoseconds
            timestamps_ps = data

            # Convert timestamps to seconds
            timestamps_s = timestamps_ps * 1e-12  # Convert to seconds

            # Save original timestamps
            original_timestamps = timestamps_s.copy()

            # Compute time differences (assuming timestamps are sorted)
            time_diffs = np.diff(timestamps_s)

            # Remove zero or negative time differences
            valid_indices = time_diffs > 0
            time_diffs = time_diffs[valid_indices]
            timestamps_filtered = timestamps_s[1:][valid_indices]

            if len(time_diffs) == 0:
                logging.warning(f"No valid time differences found in {filepath.name}")
                return None

            # Save processed results
            # Generate a unique filename to prevent overwriting
            unique_id = uuid.uuid4().hex[:8]
            output_filename = self.processed_directory / f"processed_{filepath.stem}_{unique_id}.npz"
            np.savez(output_filename,
                     timestamps=original_timestamps,
                     timestamps_filtered=timestamps_filtered,
                     time_diffs=time_diffs)

            logging.info(f"Processed {filepath.name}")
            logging.info(f"Saved variables in {output_filename.name}: {list(np.load(output_filename).keys())}")

            return output_filename  # Return the path to the processed file

        except Exception as e:
            logging.error(f"Error processing {filepath.name}: {str(e)}")
            return None

    def read_binary_file(self, filepath):
        """Read binary data from a Photon Timer .bin file"""
        try:
            # Adjust the dtype and byte order according to your .bin file format
            # Here, we assume 64-bit unsigned integers in little-endian format
            data = np.fromfile(filepath, dtype='<u8')
            return data
        except Exception as e:
            logging.error(f"Error reading binary file {filepath.name}: {str(e)}")
            return None
