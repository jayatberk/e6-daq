import numpy as np
import os
import h5py
import logging
from pathlib import Path
from typing import Optional

SAMP_FREQ = 200e6

def process_gagescope_file(filepath: Path) -> Optional[Path]:
    """
    Process an .h5 file that could be either gagescope or JKAM/High NA Imaging data.
    Logic:
    - If filename contains 'gage_shot_', treat as gagescope:
      * Load all CHx_frameY datasets (e.g., CH1_frame0, CH1_frame1, CH1_frame2, CH3_frame0, CH3_frame1, CH3_frame2)
      * Create a timestamps array based on SAMP_FREQ and dataset length.
    - If filename contains 'jkam_capture_', treat as JKAM (High NA Imaging):
      * Load all frame-xx datasets (e.g., frame-02, frame-03, etc.)
      * Create a dummy timestamps array (just a single value or a small array).
    - If no pattern found, return None.
    """

    try:
        filename = filepath.name
        with h5py.File(filepath, 'r') as hf:
            if 'gage_shot_' in filename:
                ch_datasets = [key for key in hf.keys() if key.startswith('CH')]
                if len(ch_datasets) == 0:
                    logging.warning(f"Gagescope file {filepath.name} has no CHx_frame datasets.")
                    return None

                # load channel frame datasets into dict
                data_dict = {}
                length = None
                for ds_name in ch_datasets:
                    arr = hf[ds_name][:]
                    data_dict[ds_name] = arr
                    if length is None:
                        length = arr.shape[0]

                if length is None:
                    logging.warning(f"No valid channel frames in {filepath.name}")
                    return None

                # tagging along with last one - creating timestamps array based on sampling freq
                dt = 1.0 / SAMP_FREQ
                timestamps = np.arange(length) * dt

                processed_dir = Path("processed_files")
                processed_dir.mkdir(exist_ok=True)
                output_filename = processed_dir / f"processed_{filepath.stem}.npz"
                np.savez(output_filename, timestamps=timestamps, **data_dict)
                logging.info(f"Processed gagescope file {filepath.name} saved as {output_filename.name}")
                return output_filename

            elif 'jkam_capture_' in filename:
                # just added newly - high na imaging
                frame_keys = [key for key in hf.keys() if key.startswith('frame-')]
                if len(frame_keys) == 0:
                    logging.warning(f"No frames found in JKAM file {filepath.name}")
                    return None

                frames_data = {}
                for fk in frame_keys:
                    frames_data[fk.replace('-', '_')] = hf[fk][:]

                # just creating dummy timestamp for now
                timestamps = np.array([0.0])

                processed_dir = Path("processed_files")
                processed_dir.mkdir(exist_ok=True)
                output_filename = processed_dir / f"processed_{filepath.stem}.npz"
                np.savez(output_filename, timestamps=timestamps, **frames_data)
                logging.info(f"Processed JKAM (High NA) file {filepath.name} saved as {output_filename.name}")
                return output_filename

            else:
                logging.warning(f"{filepath.name} does not match gage_shot_ or jkam_capture_ patterns.")
                return None

    except Exception as e:
        logging.error(f"Error processing h5 file {filepath.name}: {str(e)}")
        return None
