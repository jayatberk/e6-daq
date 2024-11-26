# gui.py

import threading
import os
from pathlib import Path
from queue import Empty
import tkinter as tk
from tkinter import ttk, filedialog
from pipeline_builder import FileProcessor, process_queue, FileWatcher
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mysql.connector
from mysql.connector import Error

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

        # Data for tracking plot
        self.cumulative_values = []
        self.experiment_numbers = []
        self.current_fft_data = None  # To store FFT data for plotting

        self.setup_database()  # Initialize the database connection

    def setup_processor(self):
        # Register processor types for different file extensions
        self.processor.register_processor(".bin", "photon")
        self.processor.register_processor(".txt", "fpga")
        self.processor.register_processor(".csv", "photon")
        self.processor.register_processor(".dat", "photon")
        self.processor.register_processor(".h5", "gagescope")  # Register Gagescope files

        # Set up the file watcher
        self.watch_path = "input_files"
        os.makedirs(self.watch_path, exist_ok=True)

        # Create and start the file watcher
        self.event_handler = FileWatcher(self.processor)
        self.processor.observer.schedule(self.event_handler, self.watch_path, recursive=False)
        self.processor.observer.start()

    def setup_database(self):
        """Set up the database connection."""
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                database='file_processor_db',
                user='user1',
                password='sisyphus'
            )
            if self.connection.is_connected():
                logging.info("Connected to MySQL database")
                # Ensure the table exists
                self.create_table()
        except Error as e:
            logging.error(f"Error connecting to MySQL database: {e}")
            self.connection = None

    def create_table(self):
        """Create the experiment_results table if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS experiment_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            experiment_number INT NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            accepted BOOLEAN NOT NULL,
            summary_statistics TEXT,
            processor_type VARCHAR(50),
            cumulative_value INT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor = self.connection.cursor()
        cursor.execute(create_table_query)
        self.connection.commit()
        cursor.close()

    def insert_record(self, experiment_number, file_name, accepted, summary_statistics, processor_type, cumulative_value):
        """Insert a new record into the database."""
        if self.connection is None or not self.connection.is_connected():
            logging.error("Not connected to the database. Record not inserted.")
            return
        insert_query = """
        INSERT INTO experiment_results (experiment_number, file_name, accepted, summary_statistics, processor_type, cumulative_value)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        try:
            cursor = self.connection.cursor()
            # Ensure data types are standard Python types
            data_tuple = (
                int(experiment_number),
                file_name,
                bool(accepted),
                summary_statistics,
                processor_type,
                int(cumulative_value)
            )
            cursor.execute(insert_query, data_tuple)
            self.connection.commit()
            cursor.close()
            logging.info(f"Inserted record for file {file_name} into the database.")
        except Error as e:
            logging.error(f"Error inserting record into MySQL database: {e}")

    def create_widgets(self):
        # Create a PanedWindow to allow resizing
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left frame for table
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)

        # Right frame for graphs
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=2)

        # Create the table with scrollbars
        self.create_table_widget()

        # Create the graphs in the right frame
        self.create_graphs()

        # Button to manually add files
        self.add_file_button = ttk.Button(self.root, text="Add File", command=self.add_file)
        self.add_file_button.pack(side=tk.BOTTOM, pady=5)

    def create_table_widget(self):
        # Create a frame for the table and scrollbars
        table_frame = ttk.Frame(self.left_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Create the table
        columns = ("Experiment Number", "File Name", "Accepted", "Summary Statistics")
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings')
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, anchor='w', width=100, stretch=False)

        # Adjust specific columns if needed
        self.table.column("Experiment Number", width=120, anchor='center', stretch=False)
        self.table.column("File Name", width=150, anchor='w', stretch=False)
        self.table.column("Accepted", width=80, anchor='center', stretch=False)
        self.table.column("Summary Statistics", width=600, anchor='w', stretch=False)

        # Add vertical scrollbar
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=v_scrollbar.set)

        # Add horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.table.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.table.configure(xscrollcommand=h_scrollbar.set)

        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_graphs(self):
        # Create a vertical PanedWindow for the graphs
        self.graphs_paned_window = ttk.PanedWindow(self.right_frame, orient=tk.VERTICAL)
        self.graphs_paned_window.pack(fill=tk.BOTH, expand=True)

        # Frame for the tracking plot
        tracking_frame = ttk.Frame(self.graphs_paned_window)
        self.graphs_paned_window.add(tracking_frame, weight=1)

        # Frame for the FFT plot
        fft_frame = ttk.Frame(self.graphs_paned_window)
        self.graphs_paned_window.add(fft_frame, weight=1)

        # Create the tracking plot
        self.tracking_figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.tracking_ax = self.tracking_figure.add_subplot(111)
        self.tracking_ax.set_title("Cumulative Accepted Files")
        self.tracking_ax.set_xlabel("Experiment Number")
        self.tracking_ax.set_ylabel("Cumulative Value")
        self.tracking_canvas = FigureCanvasTkAgg(self.tracking_figure, master=tracking_frame)
        self.tracking_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create the FFT plot
        self.fft_figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.fft_ax = self.fft_figure.add_subplot(111)
        self.fft_ax.set_title("FFT of the Signal")
        self.fft_ax.set_xlabel("Frequency")
        self.fft_ax.set_ylabel("Amplitude")
        self.fft_canvas = FigureCanvasTkAgg(self.fft_figure, master=fft_frame)
        self.fft_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initially hide the FFT plot
        fft_frame.pack_forget()
        self.fft_frame = fft_frame  # Keep a reference to show/hide later

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

                # Convert NumPy types to standard Python types
                accepted = bool(accepted)
                experiment_number = int(stats.get('experiment_number', 0))
                cumulative_value = int(stats.get('cumulative_value', 0))
                num_shots = int(stats.get('num_shots', 0))
                percent_space_correct = float(stats.get('percent_space_correct', 0.0))
                avg_time_gap = float(stats.get('avg_time_gap', 0.0))
                deviation_threshold_percent = float(stats.get('deviation_threshold_percent', 20.0))

                status = "Accepted" if accepted else "Rejected"
                file_name = stats.get('file_name', '')
                summary_stats = f"Space Correct: {percent_space_correct:.2f}%, " \
                                f"Avg Time Gap: {avg_time_gap:.2f}, " \
                                f"Shots: {num_shots}, " \
                                f"Threshold: {deviation_threshold_percent}%"

                # Add data to table
                self.table.insert('', tk.END, values=(experiment_number, file_name, status, summary_stats))

                # Insert record into the database
                self.insert_record(
                    experiment_number=experiment_number,
                    file_name=file_name,
                    accepted=accepted,
                    summary_statistics=summary_stats,
                    processor_type=processor_type,
                    cumulative_value=cumulative_value
                )

                # Update tracking plot data
                self.cumulative_values.append(cumulative_value)
                self.experiment_numbers.append(experiment_number)
                self.update_tracking_plot()

                # Update FFT plot data
                fft_data = stats.get('fft_data', None)
                if fft_data is not None and processor_type == "gagescope":
                    self.current_fft_data = fft_data
                    self.update_fft_plot(show=True)
                else:
                    # Hide the FFT plot if not a Gagescope file
                    self.update_fft_plot(show=False)
        except Empty:
            pass
        except Exception as e:
            logging.error(f"Error in update_display: {str(e)}")
        finally:
            self.root.after(1000, self.update_display)

    def update_tracking_plot(self):
        """Update the tracking plot with new data."""
        self.tracking_ax.clear()
        self.tracking_ax.plot(self.experiment_numbers, self.cumulative_values, marker='o')
        self.tracking_ax.set_title("Cumulative Accepted Files")
        self.tracking_ax.set_xlabel("Experiment Number")
        self.tracking_ax.set_ylabel("Cumulative Value")
        self.tracking_ax.grid(True)
        self.tracking_canvas.draw()

    def update_fft_plot(self, show=True):
        """Update the FFT plot with new data."""
        if show and self.current_fft_data is not None:
            freq = self.current_fft_data.get('freq')
            amplitude = self.current_fft_data.get('amplitude')
            if freq is not None and amplitude is not None and len(freq) > 0 and len(amplitude) > 0:
                self.fft_ax.clear()
                self.fft_ax.plot(freq, amplitude)
                self.fft_ax.set_title("FFT of the Signal")
                self.fft_ax.set_xlabel("Frequency")
                self.fft_ax.set_ylabel("Amplitude")
                self.fft_ax.grid(True)
                self.fft_canvas.draw()
                # Ensure the FFT frame is visible
                self.fft_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            else:
                logging.warning("FFT data is empty. Cannot update FFT plot.")
                self.fft_frame.pack_forget()
        else:
            # Hide the FFT plot
            self.fft_frame.pack_forget()

    def on_closing(self):
        logging.info("Shutting down...")
        if self.connection is not None and self.connection.is_connected():
            self.connection.close()
            logging.info("Database connection closed.")
        self.processor.should_continue = False
        self.processor.observer.stop()
        self.processor.observer.join()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    root = tk.Tk()
    app = GUIApp(root)
    app.run()

if __name__ == "__main__":
    main()
