import csv
import cv2
import os
import threading
import queue
import time
import logging

class CSVReader:
    def __init__(self, csv_path,queue_size=10):
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
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.csv_logger=logging.getLogger("csv_logger")
        # Start the background producer thread
        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()

    def _producer(self):
        while not self.stopped and self.index < len(self.rows):
            # If the queue is full, this will wait (preventing memory bloat)
            cpu_start = time.thread_time()
            row = self.rows[self.index]
            timestamp = float(row["#timestamp [ns]"])
            file_path = os.path.join(self.directory, row["filename"])
            
            # 1. Read and Decode (The slow parts)
            frame = cv2.imread(file_path)
            if frame is not None:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 2. Put in queue
                self.queue.put((gray_frame, timestamp))
            cpu_time = int((time.thread_time() - cpu_start) * 1000)
            #self.csv_logger.info(f" csv cpu {cpu_time:.1f}ms")
            self.index += 1

    # def read_frame(self,):
    #     try:
    #         row = self.rows[self.index]
    #     except IndexError:
    #         print("No more frames to read.")
    #         return None, None
    #     timestamp = float(row["#timestamp [ns]"])
    #     file=os.path.join(self.directory,row["filename"])
    #     frame = cv2.imread(file)
    #     self.index += 1
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     return gray_frame, timestamp
    
    def read_frame(self):
        try:
            # Pull from the pre-loaded queue
            return self.queue.get(timeout=2)
        except queue.Empty:
            return None, None

    def stop(self):
        self.stopped = True
        self.thread.join()

    def release(self):
        pass
