import os
from pathlib import Path
import threading
from queue import Queue, Empty
from typing import Dict
import logging
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from photon_processor import process_photon_file
from fpga_processor import process_fpga_file

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

        # Initialize lists to store creation times and acceptance status
        self.jkam_creation_time_array = []  # List of file creation times
        self.accepted_files = []            # List of acceptance statuses
        self.avg_time_gap = None            # Average time gap between files

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
        else:
            logging.warning(f"Unknown processor type: {processor_type}")
            return

        if output_file is None:
            logging.error(f"Failed to process file {filepath}")
            return

        # Evaluate the processed file
        accepted = self.evaluate_file(output_file, filepath)

        # Add to display queue
        self.display_queue.put((processor_type, output_file, accepted))

        logging.info(f"Finished processing: {filepath}")

    def evaluate_file(self, output_file: Path, original_file: Path) -> bool:
        """Decide whether to accept or reject the file based on time gaps between files."""
        try:
            # Get the creation time of the original file
            time_temp = os.path.getmtime(original_file)
            self.jkam_creation_time_array.append(time_temp)
            num_shots = len(self.jkam_creation_time_array)
            shot_num = num_shots - 1  # Current shot index

            # If less than 2 files, accept by default
            if num_shots < 2:
                self.avg_time_gap = None
                accepted = True
                self.accepted_files.append(accepted)
                return accepted

            # Calculate average time gap between files
            time_diffs = np.diff(self.jkam_creation_time_array)
            self.avg_time_gap = np.mean(time_diffs)

            space_correct = True

            # Check time gap with previous file
            prev_time = self.jkam_creation_time_array[shot_num - 1]
            time_gap = time_temp - prev_time
            if abs(time_gap - self.avg_time_gap) > 0.3 * self.avg_time_gap:
                space_correct = False
                logging.info(f'Error at shot number {shot_num}: Time gap deviation exceeds threshold.')

            # Decide acceptance based on space correctness
            accepted = space_correct
            self.accepted_files.append(accepted)
            return accepted

        except Exception as e:
            logging.error(f"Error evaluating file {output_file.name}: {str(e)}")
            return False


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
