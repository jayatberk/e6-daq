import threading
import os
from pathlib import Path
from queue import Empty
import tkinter as tk
from tkinter import ttk, filedialog
from pipeline_builder import FileProcessor, process_queue, FileWatcher
import logging
import numpy as np  # Ensure numpy is imported
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

        # Left frame for listbox
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Listbox to display processed files with accept/reject status and stats
        self.file_listbox = tk.Listbox(self.left_frame, height=15)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for the listbox
        self.scrollbar = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=self.scrollbar.set)

        # Right frame for FFT plot
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create a matplotlib Figure
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("FFT of the Signal")
        self.ax.set_xlabel("Frequency")
        self.ax.set_ylabel("Amplitude")

        # Embed the matplotlib Figure in Tkinter Canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
                processor_type, output_file, accepted, stats = self.processor.display_queue.get_nowait()
                status = "Accepted" if accepted else "Rejected"

                # Build the display text with relevant statistics
                if stats:
                    percent_space_correct = stats.get('percent_space_correct', 0.0)
                    avg_time_gap = stats.get('avg_time_gap', 0.0)
                    num_shots = stats.get('num_shots', 0)
                    deviation_threshold_percent = stats.get('deviation_threshold_percent', 20.0)

                    display_text = (f"{output_file.name} - {status} "
                                    f"(Space Correct: {percent_space_correct:.2f}%, "
                                    f"Avg Time Gap: {avg_time_gap:.2f}, "
                                    f"Shots: {num_shots}, "
                                    f"Threshold: {deviation_threshold_percent}%)")
                else:
                    display_text = f"{output_file.name} - {status}"

                logging.info(f"Displaying: {display_text}")
                self.file_listbox.insert(tk.END, display_text)

                # Update the FFT plot
                fft_data = stats.get('fft_data', None)
                if fft_data is not None:
                    self.update_fft_plot(fft_data)
        except Empty:
            pass
        except Exception as e:
            logging.error(f"Error in update_display: {str(e)}")
        self.root.after(1000, self.update_display)

    def update_fft_plot(self, fft_data):
        """Update the FFT plot with new data."""
        self.ax.clear()
        self.ax.plot(fft_data['freq'], fft_data['amplitude'])
        self.ax.set_title("FFT of the Signal")
        self.ax.set_xlabel("Frequency")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

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
