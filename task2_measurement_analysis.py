"""
This file aims at reading in the measurement data files in matlab .mat format.
There is an individual .mat file per yaw angle.

General Processing Steps:
1. Read in the .mat files for each yaw angle.
2. Extract the relevant measurement data from each file.
3. Process the raw extracted data (time series, frequency domain, statistical distribution, etc
    to derive meaningful metrics for further calculation
4. Store the processed data in a structured format for further analysis and visualization.
5. Generate a wake map for each yaw angle based on the processed data.

"""

import mat73

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
        self.processed_data = {}
        self.data_processing()  # Call the data processing method upon initialization

    def data_processing(self):
        """
        Processes the raw measurement data from ndarrays 01 to 07 to derive meaningful metrics for further calculation.
        This method should implement the necessary processing steps (e.g., amplitude domain analysis,
        frequency domain analysis, statistical distribution analysis) to extract relevant information
        from the raw data.

        The processed data is stored in a structured format for further analysis and visualization.
        """
        # Amplitude Domain Analysis


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