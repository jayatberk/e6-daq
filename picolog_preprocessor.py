import numpy as np
from pathlib import Path
from scipy import signal
import logging

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

            # Compute time differences (assuming timestamps are sorted)
            time_diffs = np.diff(timestamps_s)

            # Remove zero or negative time differences
            time_diffs = time_diffs[time_diffs > 0]

            if len(time_diffs) == 0:
                logging.warning(f"No valid time differences found in {filepath.name}")
                return None

            # Apply preprocessing steps
            median_time_diff = np.median(time_diffs)
            if median_time_diff <= 0:
                logging.warning(f"Invalid median time difference in {filepath.name}")
                return None

            fs = 1 / median_time_diff  # Sampling frequency estimation

            # Set a minimum fs to avoid extremely low sampling frequency
            min_fs = 1.0  # Adjust as appropriate
            if fs < min_fs:
                fs = min_fs
                logging.info(f"Sampling frequency too low, set to minimum fs={fs}")

            cutoff = 0.1  # Define cutoff frequency
            if cutoff >= fs / 2:
                cutoff = fs / 2 - fs * 0.01  # Slightly below Nyquist frequency
                logging.info(f"Cutoff frequency adjusted to {cutoff} to be below Nyquist frequency")

            filtered_data = self.butter_lowpass_filter(time_diffs, cutoff=cutoff, fs=fs, order=4)

            # Downsample data if it's too large for plotting
            max_points = 10000
            if len(filtered_data) > max_points:
                indices = np.linspace(0, len(filtered_data) - 1, max_points).astype(int)
                filtered_data = filtered_data[indices]
                timestamps_filtered = timestamps_s[1:][indices]
            else:
                timestamps_filtered = timestamps_s[1:]  # Exclude the first timestamp due to diff

            # Ensure lengths match
            min_length = min(len(filtered_data), len(timestamps_filtered))
            filtered_data = filtered_data[:min_length]
            timestamps_filtered = timestamps_filtered[:min_length]

            # Save processed results
            output_filename = self.processed_directory / f"processed_{filepath.stem}.npz"
            np.savez(output_filename,
                     timestamps=timestamps_filtered,
                     time_diffs=filtered_data)  # Save filtered time differences as 'time_diffs'
           
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


def process_photon_file(filepath: Path):
    """Process photon data using PicologPreprocessor"""
    preprocessor = PicologPreprocessor(processed_directory='./processed_photon_data')
    output_file = preprocessor.process_file(filepath)
    return output_file
