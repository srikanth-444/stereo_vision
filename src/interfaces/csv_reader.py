import csv
import cv2
import os
from pathlib import Path

class CSVReader:
    def __init__(self, csv_path):
        self.directory, filename = os.path.split(csv_path)
        print(f"CSV directory: {self.directory}, filename: {filename}")
        self.directory = os.path.join(self.directory, 'data')  # Ensure directory ends with a separator
        self.rows = []
        try:
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.rows.append(row)
            self.index = 0
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise ValueError(f"Failed to read CSV file at {csv_path}")

    def read_frame(self,):
        try:
            row = self.rows[self.index]
        except IndexError:
            print("No more frames to read.")
            return None, None
        timestamp = float(row["#timestamp [ns]"])
        file=os.path.join(self.directory,row["filename"])
        frame = cv2.imread(file)
        self.index += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return gray_frame, timestamp

    def release(self):
        pass
