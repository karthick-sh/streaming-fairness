import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
BITS_IN_BYTE = 8.0

CHUNK_DURATION = 320.0  # duration in seconds
CHUNK_JUMP = 60.0  # shift in seconds

LOWER_BW_BOUND = 1 #mbps
UPPER_BW_BOUND = 12 #mbps

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def is_artificial_trace(trace):
    return trace[-10:] == "artificial"

def standardize_chunks(data_dir, out_dir):
    files = os.listdir(data_dir)

    for trace in files:
        if is_artificial_trace(trace):
            continue
        file_path = data_dir +  trace
        output_path = out_dir + trace

        print(file_path)

        time_window = []
        bw_window = []
        with open(file_path, 'r') as f:
            for line in f:
                time_window.append(float(line.split()[0]))
                bw_window.append(float(line.split()[1]))

        time_window = np.array(time_window)

        chunk = 0
        start_time = 0
        while True:
            end_time = start_time + CHUNK_DURATION
            if start_time != 0 and end_time > np.max(time_window): 
                break

            start_ptr = find_nearest(time_window, start_time)
            end_ptr = find_nearest(time_window, end_time)

            # Check if average bw is above lower bound and below upper bound
            avg_bw = np.average(bw_window[start_ptr:end_ptr+1])
            if avg_bw < LOWER_BW_BOUND or avg_bw > UPPER_BW_BOUND:
                start_time += CHUNK_JUMP
                continue

            print("time_range: {} - {} ({})".format(start_time, end_time, avg_bw))

            with open("{}_{}".format(output_path, chunk), "w") as f:
                for i in range(start_ptr, end_ptr + 1):
                    towrite = time_window[i] - time_window[start_ptr]
                    f.write("{} {}\n".format(towrite, bw_window[i]))

            start_time += CHUNK_JUMP
            chunk += 1
        print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage python standardize_traces.py <data_dir> <out_dir>")
    standardize_chunks(sys.argv[1], sys.argv[2])
