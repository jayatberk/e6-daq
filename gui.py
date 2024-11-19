import threading
import os
from pathlib import Path
from queue import Empty
import tkinter as tk
from tkinter import ttk, filedialog
from pipeline_builder import FileProcessor, process_queue, FileWatcher
import matplotlib.pyplot as plt
import logging
import numpy as np  # Ensure numpy is imported

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Processor GUI")
        self.processor = FileProcessor()
        self.setup_processor()
        self.create_widgets()
        self.start_background_threads()
        self.update_display()

    def setup_processor(self):
        # Register processor types for different file extensions
        self.processor.register_processor(".bin", "photon")
        self.processor.register_processor(".txt", "fpga")
        self.processor.register_processor(".csv", "photon")
        self.processor.register_processor(".dat", "photon")

        # Set up the file watcher
        self.watch_path = "input_files"
        os.makedirs(self.watch_path, exist_ok=True)

        # Create and start the file watcher
        self.event_handler = FileWatcher(self.processor)
        self.processor.observer.schedule(self.event_handler, self.watch_path, recursive=False)
        self.processor.observer.start()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Listbox to display processed files with accept/reject status
        self.file_listbox = tk.Listbox(self.main_frame, height=15)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for the listbox
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=self.scrollbar.set)

        # Button to manually add files
        self.add_file_button = ttk.Button(self.root, text="Add File", command=self.add_file)
        self.add_file_button.pack(side=tk.BOTTOM, pady=5)

    def add_file(self):
        file_paths = filedialog.askopenfilenames()
        if file_paths:
            os.makedirs(self.watch_path, exist_ok=True)
            for file_path in file_paths:
                dest_path = os.path.join(self.watch_path, os.path.basename(file_path))
                # Copy the selected file to the watch directory
                with open(file_path, 'rb') as src_file, open(dest_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
                logging.info(f"File {file_path} added to watch directory.")

    def start_background_threads(self):
        # Start the processing thread
        self.processing_thread = threading.Thread(target=process_queue, args=(self.processor,), daemon=True)
        self.processing_thread.start()

    def update_display(self):
        try:
            while True:
                processor_type, output_file, accepted = self.processor.display_queue.get_nowait()
                status = "Accepted" if accepted else "Rejected"
                display_text = f"{output_file.name} - {status}"
                logging.info(f"Displaying: {display_text}")
                self.file_listbox.insert(tk.END, display_text)
                # Now attempt to display results even if rejected (GUI handles this)
                if processor_type == "fpga":
                    self.display_fpga_results(output_file, accepted)
                elif processor_type == "photon":
                    self.display_photon_results(output_file, accepted)
        except Empty:
            pass
        except Exception as e:
            logging.error(f"Error in update_display: {str(e)}")
        self.root.after(1000, self.update_display)

    def display_photon_results(self, output_file: Path, accepted: bool):
        """Display results from processed photon data"""
        data = np.load(output_file)
        keys = data.files
        if 'timestamps' not in keys or 'time_diffs' not in keys:
            return

        timestamps = data['timestamps']
        time_diffs = data['time_diffs']

        # Ensure lengths match
        min_length = min(len(timestamps), len(time_diffs))

        if min_length == 0:
            logging.warning(f"No data to plot in {output_file.name}")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps[:min_length], time_diffs[:min_length], 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('Filtered Time Differences (s)')
        title_status = "Accepted" if accepted else "Rejected"
        plt.title(f'Filtered Photon Time Differences - {title_status}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def display_fpga_results(self, output_file: Path, accepted: bool):
        """Display results from processed FPGA data"""
        data = np.load(output_file)
        data_array = data['data']
        if len(data_array) == 0:
            logging.warning(f"No data to plot in {output_file.name}")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(data_array, 'b-')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        title_status = "Accepted" if accepted else "Rejected"
        plt.title(f'Processed FPGA Data - {title_status}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def on_closing(self):
        logging.info("Shutting down...")
        self.processor.should_continue = False
        self.processor.observer.stop()
        self.processor.observer.join()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = GUIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
