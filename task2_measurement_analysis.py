"""
The experiment where the data originates from was conducted with a model wind turbine in a wind tunnel. 3 different yaw angles were tested: 
-30°, 0°, +30°. The measurement data was recorded in matlab .mat files. Each yaw angle has its own .mat file. The data was collected at a
sampling frequency of 10 kHz for a duration of 10 seconds. The data for each yaw angle was collected at 23 different measurement points in the wind tunnel
using the same sensors on a traversing system. The data was collected in the form of time series.

This file aims at reading in the measurement data files in matlab .mat format.

General Processing Steps:
1. Read in the .mat files for each yaw angle.
2. Extract the relevant measurement data from each file.
3. Process the raw extracted data (time series, frequency domain, statistical distribution, etc.)
    to derive meaningful metrics for further calculation
4. Store the processed data in a structured format for further analysis and visualization.
5. Generate a wake map for each yaw angle based on the processed data.

"""

# imports
import mat73
import numpy as np
from scipy.signal import welch, find_peaks
import matplotlib.pyplot as plt


class MatFilereader:
    """
    A class to read and process measurement data from .mat files for different yaw angles.
    Aims to fulfill steps 1 and 2 of the general processing steps outlined in the module docstring.
    """

    def __init__(self, file_paths):
        """
        Initializes the MatFilereader with a list of file paths to .mat files.

        Parameters:
        file_paths (list): A list of strings representing the paths to the .mat files.
        """
        self.file_paths = file_paths
        self.rawdata = {}

        for file_path in self.file_paths:
            mat_data = mat73.loadmat(file_path, use_attrdict=True)
            self.rawdata[file_path] = mat_data
            """
            Description of self.rawdata:
            A dictionary containing the raw measurement data for each yaw angle.
            Keys are the file paths, values are called 'resultscell'.
            'resultscell' is a list of lists of length 23. 23 is the number of individual
            measurement points that were recorded in the wind tunnel for each yaw angle. The lists
            are numbered from 00 to 22. Each list (e.g. 00) contains 2 lists (0 and 1).
            0 contains the 12 name strings: 00 = 'Name', 01 = 'Time', 02 = 'Angle', 03 = 'Angle wonan',
            04 = 'Mach', 05 = 'Velocity', 06 = 'p_t', 07 = 'p_s', 08 = 'fileid', 09 = 'density', 10 = 'coord',
            11 = 'U_WT'. 1 has length 12 and contains the actual measurement data for each of the 12 names in 0.
            01 to 07 are ndarrays containing data which was collected at the defined sampling frequency. They contain
            the time series data which has to be processed in step 3.
            """

        print("Raw measurement data successfully read from .mat files.")


class DataProcessor:
    """
    A class to process the raw measurement data extracted from .mat files.
    Aims to fulfill steps 3 and 4 of the general processing steps outlined in the module docstring.
    """

    def __init__(self, rawdata):
        """
        Initializes the DataProcessor with raw measurement data.

        Parameters:
        rawdata (dict): A dictionary containing raw measurement data for each yaw angle.
        """
        self.rawdata = rawdata
        self.processed_data = {} # A top-level dictionary to store processed data for each yaw angle
        self.sampling_frequency = 10000  # Sampling frequency (in Hz) for the time series data
        self.measuremnt_duration = 10  # Duration (in seconds) of the measurement data
        self.data_processing()  # Call the data processing method upon initialization

    def data_processing(self):
        """
        Processes the raw measurement data from ndarrays 01 to 07 to derive meaningful metrics for further calculation.
        This method should implement the necessary processing steps (e.g., amplitude domain analysis,
        frequency domain analysis, statistical distribution analysis) to extract relevant information
        from the raw data.

        The processed data is stored in a structured format for further analysis and visualization.
        """
        # Fill processed_data dictionary with yaw angle keys and empty dictionaries as values
        for file_path in self.rawdata.keys():
            yaw_angle = file_path.split('_')[-1].split('.')[0]  # Extract yaw angle from file path
            self.processed_data[yaw_angle] = {}  # Initialize an empty dictionary for each yaw angle

        # Top loop files
        for file_path, mat_data in self.rawdata.items():
            resultscell = mat_data['resultscell']
            # Dict init containing empty dicts under each measurement point key (00 to 22) for each yaw angle
            yaw_angle = file_path.split('_')[-1].split('.')[0]  # Extract yaw angle from file path
            self.processed_data[yaw_angle] = {f"{i:02d}": {} for i in range(23)}  # Initialize empty dicts for each measurement point (00 to 22)

            # Secondary loop trough the 23 measurement points (00 to 22)
            for measurement_point in resultscell:
                processed_metrics = {}  # Initialize a dict to store processed metrics for the current measurement point
                # Extract the relevant ndarrays (02 to 07) for processing
                angle_data = measurement_point[1][2]        # Assuming 02 is at index 2
                mach_data = measurement_point[1][4]         # Assuming 04 is at index 4
                velocity_data = measurement_point[1][5]     # Assuming 05 is at index 5
                p_t_data = measurement_point[1][6]          # Assuming 06 is at index 6
                p_s_data = measurement_point[1][7]          # Assuming 07 is at index 7

                # Third loop through the extracted ndarrays which contain time-series informationfor processing
                for data_array in [angle_data, mach_data, velocity_data, p_t_data, p_s_data]:
                    
                    # Perform amplitude domain analysis on all arrays (mean, median, standard deviation, min, max)
                    mean_value = data_array.mean()
                    median_value = np.median(data_array)
                    std_dev = data_array.std()
                    min_value = data_array.min()
                    max_value = data_array.max()
                    rms_value = np.sqrt(np.mean(data_array**2))  # Root Mean Square (RMS) value

                    # Assign name to the processed metrics based on the data array being processed
                    if np.array_equal(data_array, angle_data):
                        metric_name = 'Angle'
                    elif np.array_equal(data_array, mach_data):
                        metric_name = 'Mach'
                    elif np.array_equal(data_array, velocity_data):
                        metric_name = 'Velocity'
                    elif np.array_equal(data_array, p_t_data):
                        metric_name = 'p_t'
                    elif np.array_equal(data_array, p_s_data):
                        metric_name = 'p_s'

                    # Perform Fast Fourier Transform (FFT) for frequency domain analysis
                    nyquist_freq = self.sampling_frequency / 2
                    n_samples = len(data_array)
                    fft_result = np.fft.fft(data_array)
                    f = np.linspace(-nyquist_freq, nyquist_freq, n_samples)
                    f = np.fft.fftshift(f)  # Shift the zero frequency component to the center of the spectrum
                    x = 2 * fft_result / n_samples  # Normalize the FFT result

                    # Perform Power Spectral Density (PSD) analysis
                    psd_result = np.abs(x) ** 2 / 2 / np.diff(f)[0]  # Compute the Power Spectral Density

                    # Welch method for Power Spectral Density estimation
                    f_welch, psd_welch = welch(data_array, fs=self.sampling_frequency, nperseg=1024)

                    peak_signals, properties = find_peaks(psd_welch, height=np.max(psd_welch) * 0.1)  # Find peaks in the PSD

                    # Store the processed metrics in a structured format
                    processed_metrics[metric_name] = {
                        'mean': mean_value,
                        'median': median_value,
                        'std_dev': std_dev,
                        'min': min_value,
                        'max': max_value,
                        'rms': rms_value,  # Root Mean Square (RMS) value
                        'fft': fft_result, # Fast Fourier Transform result
                        'psd': psd_result,  # Power Spectral Density
                        'fwelch': f_welch,  # Frequencies for Welch method
                        'psdwelch': psd_welch,  # Power Spectral Density from Welch method
                        'peaks': peak_signals,  # Indices of peaks in the PSD
                        'peak_properties': properties  # Properties of the detected peaks
                    }

                # add the processed metrics to the corresponding measurement point in the processed_data dictionary
                measurement_point_key = f"{resultscell.index(measurement_point):02d}"  # Get the measurement point key (00 to 22)
                self.processed_data[yaw_angle][measurement_point_key] = processed_metrics  # Store the processed metrics for the current measurement point


        print("Data processing completed. Processed data is ready for further analysis.")


class WakeMapGenerator:
    """
    A class to generate wake maps based on the processed measurement data.
    Aims to fulfill step 5 of the general processing steps outlined in the module docstring.
    """

    def __init__(self, processed_data):
        """
        Initializes the WakeMapGenerator with processed measurement data.

        Parameters:
        processed_data (dict): A dictionary containing processed measurement data for each yaw angle.
        """
        self.processed_data = processed_data

# Execution
if __name__ == "__main__":

    file_paths = ["C:/dev/wind-tunnel-testing/measurements/yaw_minus30/00_Results/Results_transient_17-6-2026_FRAP_group02_yawM30.mat", 
                  "C:/dev/wind-tunnel-testing/measurements/yaw00/00_Results/Results_transient_17-6-2026_FRAP_group02_yaw00.mat",
                  "C:/dev/wind-tunnel-testing/measurements/yaw_plus30/00_Results/Results_transient_17-6-2026_FRAP_group02_yawP30.mat"]
    
    mat_reader = MatFilereader(file_paths)
    data_processor = DataProcessor(mat_reader.rawdata)
    wake_map_generator = WakeMapGenerator(data_processor.processed_data)
    plt.show()