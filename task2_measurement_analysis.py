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
import os


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
            for point_idx, measurement_point in enumerate(resultscell):
    
                processed_metrics = {}  # Initialize a dict to store processed metrics for the current measurement point
                # Extract the relevant ndarrays (02 to 07) for processing
                #angle_data = measurement_point[1][2]        # Assuming 02 is at index 2
                mach_data = measurement_point[1][4]         # Assuming 04 is at index 4
                velocity_data = measurement_point[1][5]     # Assuming 05 is at index 5
                p_t_data = measurement_point[1][6]          # Assuming 06 is at index 6
                p_s_data = measurement_point[1][7]          # Assuming 07 is at index 7
                coordinates = measurement_point[1][10]              # Assuming 10 is at index 10
                windspeed_windtunnel = measurement_point[1][11]              # Assuming 11 is at index 11

                # Third loop through the extracted ndarrays which contain time-series informationfor processing
                # for data_array in [angle_data, mach_data, velocity_data, p_t_data, p_s_data]:
                for data_array in [mach_data, velocity_data, p_t_data, p_s_data]:

                    # Perform amplitude domain analysis on all arrays (mean, median, standard deviation, min, max)
                    mean_value = data_array.mean()
                    median_value = np.median(data_array)
                    std_dev = data_array.std()
                    min_value = data_array.min()
                    max_value = data_array.max()
                    rms_value = np.sqrt(np.mean(data_array**2))  # Root Mean Square (RMS) value

                    # Assign name to the processed metrics based on the data array being processed
                    #if np.array_equal(data_array, angle_data):
                        #metric_name = 'Angle'
                    if np.array_equal(data_array, mach_data):
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
                measurement_point_key = f"{point_idx:02d}"  # Get the measurement point key (00 to 22)
                # Add the coordinates and windspeed prior to turbine to the processed metrics for the current measurement point
                processed_metrics['coordinates'] = coordinates
                processed_metrics['windspeed_windtunnel'] = windspeed_windtunnel
                self.processed_data[yaw_angle][measurement_point_key] = processed_metrics  # Store the processed metrics for the current measurement point

        # Save processed data to a .npz file for later use
        np.savez("processed_data.npz", **self.processed_data)

        print("Data processing completed and saved. Processed data is ready for further analysis.")


class WakeMapGenerator:
    """
    A class to generate wake maps based on the processed measurement data.
    Aims to fulfill step 5 of the general processing steps outlined in the module docstring.
    """

    def __init__(self, processed_data, output_directory, reference_windspeed):
        """
        Initializes the WakeMapGenerator with processed measurement data.

        Parameters:
        processed_data (dict): A dictionary containing processed measurement data for each yaw angle.
        output_directory (str): The directory where the generated wake maps will be saved.
        reference_windspeed (float): The reference wind speed used for calculating velocity deficits.
        """
        # Failfast checks
        if not processed_data:
            raise ValueError("Processed data dictionary is empty.")
        if not isinstance(output_directory, str):
            raise TypeError("output_directory must be a string.")
        if not isinstance(reference_windspeed, (int, float)):
            raise TypeError("reference_windspeed must be a number.")

        self.processed_data = processed_data
        self.output_directory = output_directory
        self.reference_windspeed = reference_windspeed

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def _extract_metric(self, yaw_angle, feature_name, metric_type):
        """
        Strict extractor for specific features and metrics from the nested dictionary.
        """
        y_coords = []
        z_coords = []
        values = []

        yaw_data = self.processed_data.get(yaw_angle)
        if yaw_data is None:
            raise KeyError(f"Yaw angle key '{yaw_angle}' missing from processed data.")

        for point_key, point_data in yaw_data.items():
            coords = point_data.get('coordinates')
            if coords is None:
                raise KeyError(f"Coordinates missing for point {point_key}")
            if len(coords) != 3:
                raise ValueError(f"Expected 3 coordinate components (x,y,z), got {len(coords)}")

            # Y is horizontal, Z is pointing down
            y_coords.append(coords[1])
            z_coords.append(coords[2])

            feature_data = point_data.get(feature_name)
            if feature_data is None:
                raise KeyError(f"Feature '{feature_name}' missing for point {point_key}")

            if metric_type == 'dominant_psd':
                psd_welch = feature_data.get('psdwelch')
                if psd_welch is None or len(psd_welch) == 0:
                    raise ValueError(f"PSD data missing or empty for {feature_name} at point {point_key}")
                values.append(np.max(psd_welch)) # Captures the dominant cyclic frequency amplitude
            else:
                val = feature_data.get(metric_type)
                if val is None:
                    raise KeyError(f"Metric '{metric_type}' missing for {feature_name} at point {point_key}")
                values.append(val)

        return np.array(y_coords), np.array(z_coords), np.array(values)

    def _calculate_wake_center(self, y_coords, z_coords, intensities):
        """
        Calculates the center of mass of the wake based on the provided intensity distribution.
        """
        if len(y_coords) != len(z_coords) or len(y_coords) != len(intensities):
            raise ValueError("Mismatched array lengths in wake center calculation.")

        # Filter out negative deficits (speed-ups outside the wake) to find the core
        weights = np.maximum(intensities, 0.0)
        total_weight = np.sum(weights)

        if total_weight <= 0:
            raise ValueError("Total weight for wake center calculation is zero or negative. No distinct wake detected.")

        y_center = np.sum(y_coords * weights) / total_weight
        z_center = np.sum(z_coords * weights) / total_weight

        return y_center, z_center

    def _plot_spatial_heatmap(self, ax, y, z, values, title, cmap, invert_z):
        """
        Helper method to plot individual spatial scatter heatmaps.
        """
        scatter = ax.scatter(y, z, c=values, cmap=cmap, s=250, edgecolors='black')
        ax.set_title(title)
        ax.set_xlabel("Y Coordinate [mm] (Horizontal Left)")
        ax.set_ylabel("Z Coordinate [mm] (Down)")
        if invert_z:
            ax.invert_yaxis() # Z points down, so higher values should be lower on the plot axis
        return scatter

    def generate_feature_analysis_heatmaps(self):
        """
        Generates comprehensive spatial heatmaps for features (Velocity, p_s, p_t)
        to visually justify which metrics are being mapped downstream.
        """
        features = ['Velocity', 'p_s', 'p_t']
        metrics = ['mean', 'rms', 'dominant_psd']
        yaw_cases = list(self.processed_data.keys())

        for feature in features:
            fig, axes = plt.subplots(len(yaw_cases), len(metrics), figsize=(18, 5 * len(yaw_cases)))
            
            # Text Transparency Requirement
            fig.suptitle(f"Feature Analysis Heatmaps: {feature}\n"
                         f"SELECTION RULE TRANSPARENCY:\n"
                         f"1. 'mean' values are selected to generate standard Time-Averaged Wake Maps.\n"
                         f"2. 'dominant_psd' values are selected to map Cyclical Wake Turbulence caused by rotor frequencies.",
                         fontsize=14, fontweight='bold', color='darkred')

            for i, yaw in enumerate(yaw_cases):
                for j, metric in enumerate(metrics):
                    ax = axes[i, j] if len(yaw_cases) > 1 else axes[j]
                    y, z, vals = self._extract_metric(yaw, feature, metric)
                    scatter = self._plot_spatial_heatmap(
                        ax, y, z, vals,
                        title=f"Yaw Case: {yaw} | Metric Map: {metric}",
                        cmap='viridis',
                        invert_z=True
                    )
                    fig.colorbar(scatter, ax=ax, label=metric)

            plt.tight_layout(rect=[0, 0, 1, 0.90])
            filepath = os.path.join(self.output_directory, f"heatmap_analysis_{feature}.png")
            plt.savefig(filepath)
            print(f"Saved feature analysis heatmap: {filepath}")

    def generate_cyclical_wake_maps(self):
        """
        Generates continuous triangulated wake maps showing both average deficit 
        and cyclic turbulence intensity, overlaying the wake center shift.
        """
        yaw_cases = list(self.processed_data.keys())

        # Setup 2 distinct figures for comparison
        fig1, axes1 = plt.subplots(1, len(yaw_cases), figsize=(7 * len(yaw_cases), 7))
        fig1.suptitle("Time-Averaged Wake Map (Velocity Deficit)\nDerived from 'mean' Velocity Feature", fontsize=16, fontweight='bold')

        fig2, axes2 = plt.subplots(1, len(yaw_cases), figsize=(7 * len(yaw_cases), 7))
        fig2.suptitle("Cyclical Wake Intensity Map (Rotor Wake Turbulence)\nDerived from 'dominant_psd' Velocity Feature", fontsize=16, fontweight='bold')

        for i, yaw in enumerate(yaw_cases):
            # --- MAP 1: Steady State Velocity Deficit ---
            y, z, mean_vel = self._extract_metric(yaw, 'Velocity', 'mean')
            # Calculate physical velocity deficit based on reference speed
            deficit = 1.0 - (mean_vel / self.reference_windspeed)

            ax1 = axes1[i]
            tricontour1 = ax1.tricontourf(y, z, deficit, levels=20, cmap='coolwarm')
            ax1.plot(y, z, 'k.', markersize=6) # Mark physical sensor nodes
            ax1.set_title(f"Yaw Case: {yaw}")
            ax1.set_xlabel("Y Coordinate [mm] (Horizontal Left)")
            ax1.set_ylabel("Z Coordinate [mm] (Down)")
            ax1.invert_yaxis()
            fig1.colorbar(tricontour1, ax=ax1, label="Velocity Deficit [-]")

            # Track and annotate wake center
            y_c1, z_c1 = self._calculate_wake_center(y, z, deficit)
            ax1.plot(y_c1, z_c1, 'r*', markersize=18, markeredgecolor='black', label="Wake Center")
            ax1.text(y_c1, z_c1 - 20, f"Wake Center:\nY={y_c1:.1f}\nZ={z_c1:.1f}", color='red', weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            # --- MAP 2: Cyclical Turbulence Intensity ---
            y_cyc, z_cyc, peak_psd = self._extract_metric(yaw, 'Velocity', 'dominant_psd')

            ax2 = axes2[i]
            tricontour2 = ax2.tricontourf(y_cyc, z_cyc, peak_psd, levels=20, cmap='plasma')
            ax2.plot(y_cyc, z_cyc, 'k.', markersize=6)
            ax2.set_title(f"Yaw Case: {yaw}")
            ax2.set_xlabel("Y Coordinate [mm] (Horizontal Left)")
            ax2.set_ylabel("Z Coordinate [mm] (Down)")
            ax2.invert_yaxis()
            fig2.colorbar(tricontour2, ax=ax2, label="Peak PSD Magnitude [(m/s)^2/Hz]")

            # Track and annotate cyclical center (often differs from velocity deficit center)
            y_c2, z_c2 = self._calculate_wake_center(y_cyc, z_cyc, peak_psd)
            ax2.plot(y_c2, z_c2, 'r*', markersize=18, markeredgecolor='black', label="Intensity Center")
            ax2.text(y_c2, z_c2 - 20, f"Intensity Center:\nY={y_c2:.1f}\nZ={z_c2:.1f}", color='red', weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        fig1.tight_layout(rect=[0, 0, 1, 0.90])
        fig2.tight_layout(rect=[0, 0, 1, 0.90])

        path1 = os.path.join(self.output_directory, "wake_map_time_averaged_deficit.png")
        path2 = os.path.join(self.output_directory, "wake_map_cyclical_intensity.png")

        fig1.savefig(path1)
        fig2.savefig(path2)
        print(f"Saved wake maps: {path1} & {path2}")

    def execute(self):
        """
        Central execution routing.
        """
        self.generate_feature_analysis_heatmaps()
        self.generate_cyclical_wake_maps()
        plt.show()

# Execution
if __name__ == "__main__":

    file_paths = ["C:/dev/wind-tunnel-testing/measurements/yaw_minus30/00_Results/Results_transient_17-6-2026_FRAP_group02_yawM30.mat", 
                  "C:/dev/wind-tunnel-testing/measurements/yaw00/00_Results/Results_transient_17-6-2026_FRAP_group02_yaw00.mat",
                  "C:/dev/wind-tunnel-testing/measurements/yaw_plus30/00_Results/Results_transient_17-6-2026_FRAP_group02_yawP30.mat"]
    
    mat_reader = MatFilereader(file_paths)
    data_processor = DataProcessor(mat_reader.rawdata)
    # Initialize generator with explicit parameters, omitting fallbacks
    wake_map_generator = WakeMapGenerator(
        processed_data=data_processor.processed_data,
        output_directory="graphs",
        reference_windspeed=6.23
    )
    
    # Execute visualization suite
    wake_map_generator.execute()
    plt.show()