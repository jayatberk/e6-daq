import numpy as np
import logging
from pathlib import Path
from picolog_preprocessor import PicologPreprocessor

def process_photon_file(filepath: Path):
    """Process photon data using PicologPreprocessor"""
    preprocessor = PicologPreprocessor(processed_directory='./processed_photon_data')
    output_file = preprocessor.process_file(filepath)
    return output_file
