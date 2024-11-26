# pipeline_builder.py

import os
from pathlib import Path
import threading
from queue import Queue, Empty
from typing import Dict, Tuple
import logging
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from photon_processor import process_photon_file
from fpga_processor import process_fpga_file
from gagescope_processor import process_gagescope_file  # Import the new processor

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FileProcessor:
    def __init__(self):
        self.processors: Dict[str, str] = {}
        self.processing_queue = Queue()
        self.display_queue = Queue()
        self.observer = Observer()
        self.should_continue = True

        # Initialize parameters for acceptance logic
        self.deviation_threshold_ratio = 0.2  # 20% deviation threshold
        self.acceptance_ratio_threshold = 80.0  # Accept if at least 80% are space_correct

        # Keep track of cumulative accepted files for plotting
        self.total_accepted_files = 0
        self.total_files_processed = 0
        self.cumulative_value = 0  # Value for the tracking plot

    def register_processor(self, extension: str, processor_type: str):
        self.processors[extension] = processor_type

    def process_file(self, filepath: Path):
        logging.info(f"Processing file: {filepath}")
        processor_type = self.processors.get(filepath.suffix)
        if not processor_type:
            logging.warning(f"No processor registered for file type: {filepath.suffix}")
            return

        # Process the file based on its type
        if processor_type == "photon":
            output_file = process_photon_file(filepath)
        elif processor_type == "fpga":
            output_file = process_fpga_file(filepath)
        elif processor_type == "gagescope":
            output_file = process_gagescope_file(filepath)
        else:
            logging.warning(f"Unknown processor type: {processor_type}")
            return

        if output_file is None:
            logging.error(f"Failed to process file {filepath}")
            return

        # Evaluate the processed file
        accepted, stats = self.evaluate_file(output_file)

        # Perform FFT and add it to stats (only for Gagescope measurements)
        if processor_type == "gagescope":
            fft_data = self.perform_fft(output_file)
            if fft_data is not None:
                stats['fft_data'] = fft_data
            else:
                logging.warning(f"FFT data not available for {output_file.name}")
        else:
            stats['fft_data'] = None  # No FFT data for other file types

        # Update cumulative value for tracking plot
        self.total_files_processed += 1
        if accepted:
            self.total_accepted_files += 1
            self.cumulative_value += 1  # Increase linearly
        else:
            self.cumulative_value = 0  # Drop to zero on rejection

        stats['cumulative_value'] = self.cumulative_value
        stats['experiment_number'] = self.total_files_processed
        stats['file_name'] = output_file.name
        stats['accepted'] = accepted
        stats['processor_type'] = processor_type  # Include processor type in stats

        # Add to display queue
        self.display_queue.put((processor_type, output_file, accepted, stats))

        logging.info(f"Finished processing: {filepath}")

    def evaluate_file(self, output_file: Path) -> Tuple[bool, Dict[str, float]]:
        """Decide whether to accept or reject the file based on data within the file."""
        try:
            data = np.load(output_file)
            if 'timestamps' not in data.files:
                logging.warning(f"No 'timestamps' found in {output_file.name}")
                return False, {}
            timestamps = data['timestamps']
            num_shots = len(timestamps)
            if num_shots < 2:
                logging.warning(f"Not enough timestamps in {output_file.name}")
                return False, {}

            # Calculate average time gap
            avg_time_gap = (timestamps[-1] - timestamps[0]) / (num_shots - 1)

            # Check space correctness for each shot
            mask_space_correct = np.ones(num_shots, dtype=bool)
            for shot_num in range(num_shots):
                time_temp = timestamps[shot_num]
                space_correct = True
                if shot_num > 0:
                    prev_time = timestamps[shot_num - 1]
                    if abs(time_temp - prev_time - avg_time_gap) > self.deviation_threshold_ratio * avg_time_gap:
                        space_correct = False
                if shot_num < (num_shots - 1):
                    next_time = timestamps[shot_num + 1]
                    if abs(-time_temp + next_time - avg_time_gap) > self.deviation_threshold_ratio * avg_time_gap:
                        space_correct = False
                mask_space_correct[shot_num] = space_correct

            # Decide acceptance based on percentage of space_correct shots
            num_space_correct = np.sum(mask_space_correct)
            percent_space_correct = (num_space_correct / num_shots) * 100

            accepted = percent_space_correct >= self.acceptance_ratio_threshold

            stats = {
                'avg_time_gap': avg_time_gap,
                'num_shots': num_shots,
                'num_space_correct': num_space_correct,
                'percent_space_correct': percent_space_correct,
                'deviation_threshold_percent': self.deviation_threshold_ratio * 100,
            }

            if accepted:
                logging.info(f"File {output_file.name} accepted: {percent_space_correct:.2f}% shots are space_correct.")
            else:
                logging.info(f"File {output_file.name} rejected: {percent_space_correct:.2f}% shots are space_correct.")

            return accepted, stats
        except Exception as e:
            logging.error(f"Error evaluating file {output_file.name}: {str(e)}")
            return False, {}

    def perform_fft(self, output_file: Path):
        """Perform FFT on the timestamps data."""
        try:
            data = np.load(output_file)
            if 'timestamps' not in data.files:
                logging.warning(f"No 'timestamps' found in {output_file.name} for FFT.")
                return None

            timestamps = data['timestamps']
            if len(timestamps) < 2:
                logging.warning(f"Not enough data for FFT in {output_file.name}.")
                return None

            # Check for monotonicity of timestamps
            if not np.all(np.diff(timestamps) > 0):
                logging.warning(f"Timestamps are not strictly increasing in {output_file.name}.")
                timestamps = np.sort(timestamps)

            # Convert timestamps to numpy array
            timestamps = np.array(timestamps)
            logging.info(f"Timestamps range: {timestamps[0]} to {timestamps[-1]}")

            # Calculate the sampling interval (assuming uniform sampling)
            time_intervals = np.diff(timestamps)
            dt = np.mean(time_intervals)
            if dt <= 0:
                logging.error(f"Invalid sampling interval dt={dt} in {output_file.name}.")
                return None

            logging.info(f"Average sampling interval dt: {dt}")

            # Interpolate timestamps to create a uniformly sampled signal
            min_time = timestamps[0]
            max_time = timestamps[-1]
            num_samples = int((max_time - min_time) / dt)
            if num_samples <= 0:
                logging.error(f"Invalid number of samples num_samples={num_samples} in {output_file.name}.")
                return None

            logging.info(f"Number of samples for FFT: {num_samples}")

            uniform_times = np.linspace(min_time, max_time, num_samples)
            signal = np.interp(uniform_times, timestamps, np.ones_like(timestamps))

            # Perform FFT
            fft_values = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(len(signal), d=dt)

            # Only keep the positive frequencies
            positive_freqs = fft_freq > 0
            fft_freq = fft_freq[positive_freqs]
            fft_values = np.abs(fft_values[positive_freqs])

            # Return FFT data
            fft_data = {'freq': fft_freq, 'amplitude': fft_values}
            return fft_data
        except Exception as e:
            logging.error(f"Error performing FFT on {output_file.name}: {str(e)}")
            return None

class FileWatcher(FileSystemEventHandler):
    def __init__(self, processor: FileProcessor):
        super().__init__()
        self.processor = processor

    def on_created(self, event):
        if not event.is_directory:
            filepath = Path(event.src_path)
            if filepath.suffix in self.processor.processors:
                # Add a slight delay to ensure the file is fully written
                threading.Timer(0.5, self.processor.processing_queue.put, args=[filepath]).start()

def process_queue(processor: FileProcessor):
    """Worker function to process files from the queue"""
    while processor.should_continue:
        try:
            filepath = processor.processing_queue.get(timeout=1)
            processor.process_file(filepath)
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Error in process_queue: {str(e)}")
