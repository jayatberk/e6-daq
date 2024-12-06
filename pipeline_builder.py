import os
import threading
import pickle 
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Tuple
import logging
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from photon_processor import process_photon_file
from fpga_processor import process_fpga_file
from gagescope_processor import process_gagescope_file
import datetime
import re

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = Path("C:\\Users\\jayom\\Downloads\\run_analyses_files")
JKAM_COUNTS_FILE = DATA_DIR / "run4_jkam_counts_array(3).pkl"
JKAM_FRAMES_FILE = DATA_DIR / "run4_jkam_frames_array(2).pkl"
PT_TIMESTAMP_FILE = DATA_DIR / "run4_FPGA_timestamp_array_raw.pkl"
GAGE_CMPLEX_FILE = DATA_DIR / "run4_hann_gage_cmplx_amp_5_1.pkl"
GAGE_TIMEBIN_FILE = DATA_DIR / "run4_hann_gage_timebin_5_1.pkl"  # Needed?

def extract_shot_num_from_filename(filename: str) -> int:
    match = re.search(r'_(\d{5})', filename)
    if match:
        return int(match.group(1))
    else:
        logging.warning(f"Could not extract shot number from filename: {filename}")
        return -1

class FileProcessor:
    def __init__(self):
        self.processors: Dict[str, str] = {}
        self.processing_queue = Queue()
        self.display_queue = Queue()
        self.observer = Observer()
        self.should_continue = True

        self.jkam_data_loaded = False
        self.jkam_counts_array = None
        self.jkam_frames_array = None
        self.pt_timestamp_array = None
        self.load_all_data()

        self.total_accepted_files = 0
        self.total_files_processed = 0
        self.cumulative_value = 0

    def load_all_data(self):
        if not JKAM_COUNTS_FILE.exists():
            logging.error(f"JKAM counts file not found: {JKAM_COUNTS_FILE}")
            return
        if not JKAM_FRAMES_FILE.exists():
            logging.error(f"JKAM frames file not found: {JKAM_FRAMES_FILE}")
            return
        try:
            with open(JKAM_COUNTS_FILE, 'rb') as f:
                self.jkam_counts_array = np.load(f, allow_pickle=True)
        except:
            with open(JKAM_COUNTS_FILE, 'rb') as f:
                self.jkam_counts_array = pickle.load(f)

        try:
            with open(JKAM_FRAMES_FILE, 'rb') as f:
                self.jkam_frames_array = np.load(f, allow_pickle=True)
        except:
            with open(JKAM_FRAMES_FILE, 'rb') as f:
                self.jkam_frames_array = pickle.load(f)

        if not isinstance(self.jkam_counts_array, np.ndarray):
            self.jkam_counts_array = np.array(self.jkam_counts_array)
        if not isinstance(self.jkam_frames_array, np.ndarray):
            self.jkam_frames_array = np.array(self.jkam_frames_array)

        self.jkam_data_loaded = True
        logging.info(f"Loaded JKAM data: counts shape={self.jkam_counts_array.shape}, frames shape={self.jkam_frames_array.shape}")

        if PT_TIMESTAMP_FILE.exists():
            try:
                with open(PT_TIMESTAMP_FILE, 'rb') as f:
                    self.pt_timestamp_array = pickle.load(f)
                logging.info("PT timestamp data loaded for JKAM timing reference.")
            except Exception as e:
                logging.warning(f"Error loading PT timestamps: {e}")
                self.pt_timestamp_array = None
        else:
            logging.warning("PT timestamp file not found, JKAM time will be N/A")
            self.pt_timestamp_array = None

    def register_processor(self, extension: str, processor_type: str):
        self.processors[extension] = processor_type

    def process_file(self, filepath: Path):
        logging.info(f"Processing file: {filepath}")
        processor_type = self.processors.get(filepath.suffix)
        if not processor_type:
            logging.warning(f"No processor registered for {filepath.suffix}")
            return

        if processor_type == "photon":
            output_file = process_photon_file(filepath)
        elif processor_type == "fpga":
            output_file = process_fpga_file(filepath)
        elif processor_type == "gagescope":
            output_file = process_gagescope_file(filepath)
        else:
            output_file = None

        if output_file is None:
            logging.error(f"Failed to process file {filepath}")
            return

        accepted, stats = self.evaluate_file(output_file, processor_type)

        if processor_type == "gagescope":
            fft_data = self.perform_fft(output_file)
            stats['fft_data'] = fft_data
        else:
            stats['fft_data'] = None

        self.total_files_processed += 1
        if accepted:
            self.total_accepted_files += 1
            self.cumulative_value += 1
        else:
            self.cumulative_value = 0

        stats['cumulative_value'] = self.cumulative_value
        stats['experiment_number'] = self.total_files_processed
        stats['file_name'] = output_file.name
        stats['accepted'] = accepted
        stats['processor_type'] = processor_type

        self.display_queue.put((processor_type, output_file, accepted, stats))
        logging.info(f"Finished processing: {filepath}, accepted={accepted}")

    def evaluate_file(self, output_file: Path, processor_type: str) -> Tuple[bool, Dict[str, float]]:
        try:
            file_creation_time = os.path.getmtime(output_file)
            shot_num = extract_shot_num_from_filename(output_file.name)

            if not self.jkam_data_loaded:
                accepted = True
                space_correct = True
                jk_shot = "N/A"
                jk_time = "N/A"
            else:
                max_shots = self.jkam_counts_array.shape[0]
                if shot_num < 0:
                    accepted = True
                    space_correct = True
                    jk_shot = "N/A"
                    jk_time = "N/A"
                else:
                    if 0 <= shot_num < max_shots:
                        accepted = True
                        space_correct = True
                        jk_shot = str(shot_num)
                        if (self.pt_timestamp_array is not None 
                            and shot_num < len(self.pt_timestamp_array) 
                            and self.pt_timestamp_array[shot_num] is not None 
                            and len(self.pt_timestamp_array[shot_num]) > 0):
                            jk_time_val = np.median(self.pt_timestamp_array[shot_num])
                            jk_time = f"{jk_time_val:.2f}"
                        else:
                            jk_time = "N/A"
                    else:
                        accepted = False
                        space_correct = False
                        jk_shot = f"{shot_num} (Invalid)"
                        jk_time = "N/A"

            summary_statistics = f"File Time: {file_creation_time:.2f}, JKAM Shot: {jk_shot}, JKAM Time: {jk_time}"

            stats = {
                'file_creation_time': file_creation_time,
                'matched_jkam_shot': jk_shot,
                'space_correct': space_correct,
                'summary_statistics': summary_statistics
            }

            if accepted:
                logging.info(f"File {output_file.name} accepted (JKAM Shot: {jk_shot}, JKAM Time: {jk_time}).")
            else:
                logging.info(f"File {output_file.name} rejected (JKAM Shot: {jk_shot}).")

            return accepted, stats

        except Exception as e:
            logging.error(f"Error evaluating file {output_file.name}: {str(e)}")
            return False, {}

    def perform_fft(self, output_file: Path):
        try:
            data = np.load(output_file)
            if 'timestamps' not in data.files:
                logging.warning(f"No 'timestamps' in {output_file.name} for FFT.")
                return None

            timestamps = data['timestamps']
            if len(timestamps) < 2:
                logging.warning("Not enough data for FFT.")
                return None

            if not np.all(np.diff(timestamps) > 0):
                logging.warning("Timestamps not strictly increasing, sorting.")
                timestamps = np.sort(timestamps)

            time_intervals = np.diff(timestamps)
            dt = np.mean(time_intervals)
            if dt <= 0:
                logging.error(f"Invalid dt={dt}")
                return None

            min_time = timestamps[0]
            max_time = timestamps[-1]
            num_samples = int((max_time - min_time) / dt)
            if num_samples <= 0:
                logging.error("No valid samples for FFT.")
                return None

            uniform_times = np.linspace(min_time, max_time, num_samples)
            signal = np.interp(uniform_times, timestamps, np.ones_like(timestamps))

            fft_values = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(len(signal), d=dt)
            positive_freqs = fft_freq > 0
            fft_freq = fft_freq[positive_freqs]
            fft_values = np.abs(fft_values[positive_freqs])

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
                logging.info(f"File {filepath} created, adding to queue.")
                threading.Timer(0.5, self.processor.processing_queue.put, args=[filepath]).start()

def process_queue(processor: FileProcessor):
    while processor.should_continue:
        try:
            filepath = processor.processing_queue.get(timeout=1)
            processor.process_file(filepath)
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Error in process_queue: {str(e)}")
