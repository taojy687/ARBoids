import numpy as np
import os

class NPZLogger:
    def __init__(self, filename):
        """
        Initialize NPZ Data Logger
        :param filename: .npz filename to save
        """
        self.filename = filename
        self.data = self._load_existing_data()

    def _load_existing_data(self):
        """Load existing data (if any)"""
        if os.path.exists(self.filename):
            with np.load(self.filename, allow_pickle=False) as loaded:
                return {key: loaded[key] for key in loaded.files}
        else:
            return {}

    def log(self, new_data_dict):
        """
        Log new data, appending values for each key to existing arrays
        :param new_data_dict: dict, key -> np.array (single data entry)
        """
        for key, array in new_data_dict.items():
            array = np.asarray(array)
            array = array[np.newaxis, ...] if array.ndim == 1 else array  # (D,) → (1, D)

            if key in self.data:
                self.data[key] = np.vstack([self.data[key], array])
            else:
                self.data[key] = array

        np.savez(self.filename, **self.data)
        print(f"[NPZLogger] Appended data for keys: {list(new_data_dict.keys())}")

    def list_keys(self):
        """Return all currently saved keys"""
        return list(self.data.keys())

    def load(self):
        """Return all currently saved data"""
        return self.data
    
