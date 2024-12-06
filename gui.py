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

        self.cumulative_values = []
        self.experiment_numbers = []
        self.current_fft_data = None

        self.setup_database()

    def setup_processor(self):
        self.processor.register_processor(".bin", "photon")
        self.processor.register_processor(".txt", "fpga")
        self.processor.register_processor(".csv", "photon")
        self.processor.register_processor(".dat", "photon")
        self.processor.register_processor(".h5", "gagescope")

        self.watch_path = "input_files"
        os.makedirs(self.watch_path, exist_ok=True)

        self.event_handler = FileWatcher(self.processor)
        self.processor.observer.schedule(self.event_handler, self.watch_path, recursive=False)
        self.processor.observer.start()

    def setup_database(self):
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                database='file_processor_db',
                user='user1',
                password='sisyphus'
            )
            if self.connection.is_connected():
                logging.info("Connected to MySQL database")
                self.create_table()
        except Error as e:
            logging.error(f"Error connecting to MySQL database: {e}")
            self.connection = None

    def create_table(self):
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
        if self.connection is None or not self.connection.is_connected():
            logging.error("Not connected to the database. Record not inserted.")
            return
        insert_query = """
        INSERT INTO experiment_results (experiment_number, file_name, accepted, summary_statistics, processor_type, cumulative_value)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        try:
            cursor = self.connection.cursor()
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
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)

        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=2)

        self.create_table_widget()
        self.create_graphs()

        self.add_file_button = ttk.Button(self.root, text="Add File", command=self.add_file)
        self.add_file_button.pack(side=tk.BOTTOM, pady=5)

    def create_table_widget(self):
        table_frame = ttk.Frame(self.left_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("Experiment Number", "File Name", "Accepted", "JKAM Shot", "Space Correct", "Summary Statistics")
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings')
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, anchor='w', width=100, stretch=False)

        self.table.column("Experiment Number", width=120, anchor='center', stretch=False)
        self.table.column("File Name", width=150, anchor='w', stretch=False)
        self.table.column("Accepted", width=80, anchor='center', stretch=False)
        self.table.column("JKAM Shot", width=100, anchor='center', stretch=False)
        self.table.column("Space Correct", width=100, anchor='center', stretch=False)
        self.table.column("Summary Statistics", width=600, anchor='w', stretch=False)

        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.table.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.table.configure(xscrollcommand=h_scrollbar.set)

        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_graphs(self):
        self.graphs_paned_window = ttk.PanedWindow(self.right_frame, orient=tk.VERTICAL)
        self.graphs_paned_window.pack(fill=tk.BOTH, expand=True)

        tracking_frame = ttk.Frame(self.graphs_paned_window)
        self.graphs_paned_window.add(tracking_frame, weight=1)

        fft_frame = ttk.Frame(self.graphs_paned_window)
        self.graphs_paned_window.add(fft_frame, weight=1)

        self.tracking_figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.tracking_ax = self.tracking_figure.add_subplot(111)
        self.tracking_ax.set_title("Cumulative Accepted Files")
        self.tracking_ax.set_xlabel("Experiment Number")
        self.tracking_ax.set_ylabel("Cumulative Value")
        self.tracking_canvas = FigureCanvasTkAgg(self.tracking_figure, master=tracking_frame)
        self.tracking_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fft_figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.fft_ax = self.fft_figure.add_subplot(111)
        self.fft_ax.set_title("FFT of the Signal")
        self.fft_ax.set_xlabel("Frequency")
        self.fft_ax.set_ylabel("Amplitude")
        self.fft_canvas = FigureCanvasTkAgg(self.fft_figure, master=fft_frame)
        self.fft_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fft_frame.pack_forget()
        self.fft_frame = fft_frame

    def add_file(self):
        file_paths = filedialog.askopenfilenames()
        if file_paths:
            os.makedirs(self.watch_path, exist_ok=True)
            for file_path in file_paths:
                dest_path = os.path.join(self.watch_path, os.path.basename(file_path))
                try:
                    with open(file_path, 'rb') as src_file, open(dest_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())
                    logging.info(f"File {file_path} added to watch directory.")
                except Exception as e:
                    logging.error(f"Error adding file {file_path}: {e}")

    def start_background_threads(self):
        self.processing_thread = threading.Thread(target=process_queue, args=(self.processor,), daemon=True)
        self.processing_thread.start()

    def update_display(self):
        try:
            while True:
                processor_type, output_file, accepted, stats = self.processor.display_queue.get_nowait()

                accepted = bool(accepted)
                experiment_number = int(stats.get('experiment_number', 0))
                cumulative_value = int(stats.get('cumulative_value', 0))
                space_correct = str(stats.get('space_correct', False))
                matched_jkam_shot = stats.get('matched_jkam_shot', "N/A")
                summary_stats = stats.get('summary_statistics', "N/A")

                status = "Accepted" if accepted else "Rejected"
                file_name = stats.get('file_name', '')

                self.table.insert('', tk.END, values=(
                    experiment_number, file_name, status, matched_jkam_shot, space_correct, summary_stats
                ))

                self.insert_record(
                    experiment_number=experiment_number,
                    file_name=file_name,
                    accepted=accepted,
                    summary_statistics=summary_stats,
                    processor_type=processor_type,
                    cumulative_value=cumulative_value
                )

                self.cumulative_values.append(cumulative_value)
                self.experiment_numbers.append(experiment_number)
                self.update_tracking_plot()

                fft_data = stats.get('fft_data', None)
                if fft_data is not None and processor_type == "gagescope":
                    self.current_fft_data = fft_data
                    self.update_fft_plot(show=True)
                else:
                    self.update_fft_plot(show=False)
        except Empty:
            pass
        except Exception as e:
            logging.error(f"Error in update_display: {str(e)}")
        finally:
            self.root.after(1000, self.update_display)

    def update_tracking_plot(self):
        self.tracking_ax.clear()
        self.tracking_ax.plot(self.experiment_numbers, self.cumulative_values, marker='o')
        self.tracking_ax.set_title("Cumulative Accepted Files")
        self.tracking_ax.set_xlabel("Experiment Number")
        self.tracking_ax.set_ylabel("Cumulative Value")
        self.tracking_ax.grid(True)
        self.tracking_canvas.draw()

    def update_fft_plot(self, show=True):
        self.fft_ax.clear()
        if show and self.current_fft_data is not None:
            freq = self.current_fft_data.get('freq')
            amplitude = self.current_fft_data.get('amplitude')
            if freq is not None and amplitude is not None and len(freq) > 0 and len(amplitude) > 0:
                self.fft_ax.plot(freq, amplitude)
                self.fft_ax.set_title("FFT of the Signal")
                self.fft_ax.set_xlabel("Frequency")
                self.fft_ax.set_ylabel("Amplitude")
                self.fft_ax.grid(True)
            else:
                self.fft_ax.text(0.5, 0.5, "No FFT data available", ha='center', va='center')
        else:
            # Just show a placeholder message if no FFT data
            self.fft_ax.text(0.5, 0.5, "No FFT data available", ha='center', va='center')
        # Was doing wrong - basically don't pack_forget() the frame, just keep it as is
        self.fft_canvas.draw()

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
