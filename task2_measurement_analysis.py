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
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, find_peaks
import matplotlib.pyplot as plt
import os
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
            if mat73 is not None:
                mat_data = mat73.loadmat(file_path, use_attrdict=True)
            # else:
            #     mat_data = loadmat(file_path, squeeze_me=False, struct_as_record=False)
            else:
                raise ImportError("mat73 module is not available. Please install it to read .mat files.")
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
        self.quality_assessment = {}
        self.sampling_frequency = 10000  # Sampling frequency (in Hz) for the time series data
        self.measuremnt_duration = 10  # Duration (in seconds) of the measurement data
        self.data_processing()  # Call the data processing method upon initialization

    @staticmethod
    def _normalize_yaw_label(file_path):
        """Map a file name to the canonical yaw label used in downstream code."""
        yaw_name = os.path.basename(file_path)
        if "yawM30" in yaw_name:
            return "-30"
        if "yaw00" in yaw_name or "yaw_00" in yaw_name:
            return "0"
        if "yawP30" in yaw_name:
            return "+30"
        return os.path.splitext(yaw_name)[0]

    @staticmethod
    def _ensure_1d_signal(raw_signal, signal_name):
        """Convert a raw MATLAB payload to a 1D floating-point NumPy array."""
        signal_array = np.asarray(raw_signal, dtype=float).squeeze()
        if signal_array.ndim != 1:
            raise ValueError(
                f"{signal_name} must resolve to a 1D time series, got shape {signal_array.shape}."
            )
        if signal_array.size == 0:
            raise ValueError(f"{signal_name} time series is empty.")
        return signal_array

    @staticmethod
    def _extract_angle_components(angle_raw):
        """Extract alpha and beta from the Angle payload.

        The institute notes say alpha is stored in column 3 and beta in column 4.
        This helper accepts either row-major or column-major MATLAB payloads.
        """
        angle_array = np.asarray(angle_raw, dtype=float)
        if angle_array.ndim != 2:
            raise ValueError(f"Angle payload must be 2D, got shape {angle_array.shape}.")

        if angle_array.shape[1] >= 4:
            alpha = np.asarray(angle_array[:, 2], dtype=float).squeeze()
            beta = np.asarray(angle_array[:, 3], dtype=float).squeeze()
        elif angle_array.shape[0] >= 4:
            alpha = np.asarray(angle_array[2, :], dtype=float).squeeze()
            beta = np.asarray(angle_array[3, :], dtype=float).squeeze()
        else:
            raise ValueError(
                f"Angle payload does not contain columns 3 and 4: shape {angle_array.shape}."
            )

        if alpha.ndim != 1 or beta.ndim != 1:
            raise ValueError("Alpha and beta must each resolve to 1D time series.")
        if alpha.size == 0 or beta.size == 0:
            raise ValueError("Alpha or beta time series is empty.")
        if alpha.shape != beta.shape:
            raise ValueError(
                f"Alpha and beta must have the same length, got {alpha.shape} and {beta.shape}."
            )

        return alpha, beta

    @staticmethod
    def _series_status(array_value):
        """Return shape, NaN flag, and inf flag for any NumPy-compatible value."""
        array_value = np.asarray(array_value)
        has_nan = np.issubdtype(array_value.dtype, np.number) and np.isnan(array_value).any()
        has_inf = np.issubdtype(array_value.dtype, np.number) and np.isinf(array_value).any()
        return array_value.shape, bool(has_nan), bool(has_inf)

    def _validate_shape(self, yaw_angle, point_idx, series_name, actual_value, expected_shape):
        """Fail fast when a time-series shape does not match the expected shape."""
        actual_shape = np.asarray(actual_value).shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch at yaw {yaw_angle}, point {point_idx:02d}, series '{series_name}': "
                f"expected {expected_shape}, got {actual_shape}."
            )

    @staticmethod
    def _count_nan(actual_value):
        """Count NaN values in any numeric NumPy-compatible payload."""
        arr = np.asarray(actual_value)
        if not np.issubdtype(arr.dtype, np.number):
            return 0
        return int(np.isnan(arr).sum())

    @staticmethod
    def _count_inf(actual_value):
        """Count +-inf values in any numeric NumPy-compatible payload."""
        arr = np.asarray(actual_value)
        if not np.issubdtype(arr.dtype, np.number):
            return 0
        return int(np.isinf(arr).sum())

    def _register_quality_counts(self, yaw_angle, point_idx, nan_before, nan_after, inf_persistent):
        """Accumulate quality counts per yaw and measurement point."""
        if yaw_angle not in self.quality_assessment:
            self.quality_assessment[yaw_angle] = {
                'nan_before': 0,
                'nan_after': 0,
                'inf_persistent': 0,
                'points': {}
            }

        point_key = f"{point_idx:02d}"
        self.quality_assessment[yaw_angle]['points'][point_key] = {
            'nan_before': int(nan_before),
            'nan_after': int(nan_after),
            'inf_persistent': int(inf_persistent),
        }

        self.quality_assessment[yaw_angle]['nan_before'] += int(nan_before)
        self.quality_assessment[yaw_angle]['nan_after'] += int(nan_after)
        self.quality_assessment[yaw_angle]['inf_persistent'] += int(inf_persistent)

    @staticmethod
    def _max_consecutive_true(mask):
        """Return the longest run of True values in a boolean mask."""
        if mask.size == 0:
            return 0
        padded = np.concatenate(([False], mask, [False]))
        changes = np.diff(padded.astype(int))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]
        if run_starts.size == 0:
            return 0
        return int(np.max(run_ends - run_starts))

    def _interpolate_nans_limited(self, signal_array, yaw_angle, point_idx, series_name, max_consecutive=100):
        """Linearly interpolate NaNs if each NaN run is within the allowed run length.

        max_consecutive=100 corresponds to 0.01 s at 10 kHz.
        """
        interpolated = np.asarray(signal_array, dtype=float).copy()
        nan_mask = np.isnan(interpolated)

        if not nan_mask.any():
            return interpolated

        longest_run = self._max_consecutive_true(nan_mask)
        if longest_run > max_consecutive:
            raise ValueError(
                f"Interpolation limit exceeded for {series_name} at yaw {yaw_angle}, point {point_idx:02d}: "
                f"longest NaN run={longest_run} samples (> {max_consecutive})."
            )

        valid_idx = np.where(~nan_mask)[0]
        if valid_idx.size < 2:
            raise ValueError(
                f"Not enough valid samples to interpolate {series_name} at yaw {yaw_angle}, point {point_idx:02d}."
            )

        nan_idx = np.where(nan_mask)[0]
        interpolated[nan_idx] = np.interp(nan_idx, valid_idx, interpolated[valid_idx])
        return interpolated

    def _print_quality_assessment(self):
        """Print only NaN before/after interpolation and persistent inf counts."""
        print("\n=== Interpolation Quality Summary ===")
        for yaw_angle in sorted(self.quality_assessment.keys()):
            yaw_data = self.quality_assessment[yaw_angle]
            print(
                f"yaw={yaw_angle}: nan_before={yaw_data['nan_before']}, "
                f"nan_after={yaw_data['nan_after']}, inf_persistent={yaw_data['inf_persistent']}"
            )

            for point_key in sorted(yaw_data['points'].keys()):
                point_data = yaw_data['points'][point_key]
                print(
                    f"  point={point_key}: nan_before={point_data['nan_before']}, "
                    f"nan_after={point_data['nan_after']}, inf_persistent={point_data['inf_persistent']}"
                )

    def _build_signal_metrics(self, signal_array, yaw_angle=None, point_idx=None, series_name=None):
        """Calculate statistical and frequency-domain metrics for one time series."""
        mean_value = signal_array.mean()
        median_value = np.median(signal_array)
        std_dev = signal_array.std()
        min_value = signal_array.min()
        max_value = signal_array.max()
        rms_value = np.sqrt(np.mean(signal_array**2))

        nyquist_freq = self.sampling_frequency / 2
        n_samples = len(signal_array)
        fft_result = np.fft.fft(signal_array)
        frequencies = np.linspace(-nyquist_freq, nyquist_freq, n_samples)
        frequencies = np.fft.fftshift(frequencies)
        normalized_fft = 2 * fft_result / n_samples

        psd_result = np.abs(normalized_fft) ** 2 / 2 / np.diff(frequencies)[0]

        f_welch, psd_welch = welch(signal_array, fs=self.sampling_frequency, nperseg=1024)
        peak_signals, properties = find_peaks(psd_welch, height=np.max(psd_welch) * 0.1)

        metrics = {
            'series': signal_array,
            'mean': np.array([mean_value]),
            'median': np.array([median_value]),
            'std_dev': np.array([std_dev]),
            'min': np.array([min_value]),
            'max': np.array([max_value]),
            'rms': np.array([rms_value]),
            'fft': fft_result,
            'psd': psd_result,
            'fwelch': f_welch,
            'psdwelch': psd_welch,
            'peaks': peak_signals,
            'peak_properties': properties,
        }

        if yaw_angle is not None and point_idx is not None and series_name is not None:
            for metric_name in {'mean', 'median', 'std_dev', 'min', 'max', 'rms'}:
                self._validate_shape(
                    yaw_angle,
                    point_idx,
                    f"{series_name}.{metric_name}",
                    metrics[metric_name],
                    (1,),
                )

        return metrics

    def data_processing(self):
        """
        Processes the raw measurement data to derive the velocity components requested by the
        wind tunnel institute.

        The calculation uses the Angle payload stored at point_payload[2]. Its columns 3 and 4
        are interpreted as alpha and beta, together with the measured velocity magnitude. The
        derived velocity components are calculated as:

        u_uncor = U * cos(alpha) * cos(beta)
        v_uncor = U * sin(beta)
        w_uncor = U * sin(alpha) * cos(beta)

        The FRAP probe orientation is then corrected with:

        u_trans = u_uncor
        v_trans = -w_uncor
        w_trans = v_uncor

        The processed data is stored in a structured format for further analysis and visualization.
        """
        self.processed_data = {}
        self.quality_assessment = {}

        for file_path, mat_data in self.rawdata.items():
            yaw_angle = self._normalize_yaw_label(file_path)
            self.processed_data[yaw_angle] = {f"{i:02d}": {} for i in range(23)}
            resultscell = mat_data['resultscell']

            for point_idx, measurement_point in enumerate(resultscell):
                point_payload = measurement_point[1]

                time_raw = point_payload[1]
                angle_raw = point_payload[2]
                velocity_raw = point_payload[5]
                coord_raw = point_payload[10]
                u_wt_raw = point_payload[11]

                time_data = self._ensure_1d_signal(time_raw, "time")
                self._validate_shape(yaw_angle, point_idx, 'time', time_data, (100000,))
                self._validate_shape(yaw_angle, point_idx, 'angle', angle_raw, (100000, 4))
                self._validate_shape(yaw_angle, point_idx, 'velocity', velocity_raw, (100000,))

                coord_data = np.asarray(coord_raw, dtype=float).reshape(-1)
                u_wt_data = np.asarray(u_wt_raw, dtype=float).reshape(-1)
                self._validate_shape(yaw_angle, point_idx, 'coord', coord_data, (3,))
                self._validate_shape(yaw_angle, point_idx, 'U_WT', u_wt_data, (1,))

                alpha_data, beta_data = self._extract_angle_components(angle_raw)
                velocity_data = self._ensure_1d_signal(velocity_raw, "velocity")

                nan_before = (
                    self._count_nan(alpha_data)
                    + self._count_nan(beta_data)
                    + self._count_nan(velocity_data)
                )

                alpha_data = self._interpolate_nans_limited(alpha_data, yaw_angle, point_idx, 'alpha', max_consecutive=100)
                beta_data = self._interpolate_nans_limited(beta_data, yaw_angle, point_idx, 'beta', max_consecutive=100)
                velocity_data = self._interpolate_nans_limited(velocity_data, yaw_angle, point_idx, 'velocity', max_consecutive=100)

                nan_after = (
                    self._count_nan(alpha_data)
                    + self._count_nan(beta_data)
                    + self._count_nan(velocity_data)
                )

                self._validate_shape(yaw_angle, point_idx, 'alpha', alpha_data, (100000,))
                self._validate_shape(yaw_angle, point_idx, 'beta', beta_data, (100000,))
                self._validate_shape(yaw_angle, point_idx, 'velocity', velocity_data, (100000,))

                if alpha_data.shape != velocity_data.shape or beta_data.shape != velocity_data.shape:
                    raise ValueError(
                        f"Shape mismatch at yaw {yaw_angle}, point {point_idx:02d}, series 'alpha/beta/velocity': "
                        f"alpha={alpha_data.shape}, beta={beta_data.shape}, velocity={velocity_data.shape}."
                    )

                alpha_rad = np.deg2rad(alpha_data)
                beta_rad = np.deg2rad(beta_data)

                u_uncor = velocity_data * np.cos(alpha_rad) * np.cos(beta_rad)
                v_uncor = velocity_data * np.sin(beta_rad)
                w_uncor = velocity_data * np.sin(alpha_rad) * np.cos(beta_rad)

                v_trans = -w_uncor
                w_trans = v_uncor

                self._validate_shape(yaw_angle, point_idx, 'u_trans', u_uncor, (100000,))
                self._validate_shape(yaw_angle, point_idx, 'v_trans', v_trans, (100000,))
                self._validate_shape(yaw_angle, point_idx, 'w_trans', w_trans, (100000,))

                inf_persistent = (
                    self._count_inf(alpha_data)
                    + self._count_inf(beta_data)
                    + self._count_inf(velocity_data)
                    + self._count_inf(u_uncor)
                    + self._count_inf(v_trans)
                    + self._count_inf(w_trans)
                )
                self._register_quality_counts(yaw_angle, point_idx, nan_before, nan_after, inf_persistent)

                processed_metrics = {
                    'time': time_data,
                    'alpha': self._build_signal_metrics(alpha_data, yaw_angle, point_idx, 'alpha'),
                    'beta': self._build_signal_metrics(beta_data, yaw_angle, point_idx, 'beta'),
                    'velocity': self._build_signal_metrics(velocity_data, yaw_angle, point_idx, 'velocity'),
                    'u_trans': self._build_signal_metrics(u_uncor, yaw_angle, point_idx, 'u_trans'),
                    'v_trans': self._build_signal_metrics(v_trans, yaw_angle, point_idx, 'v_trans'),
                    'w_trans': self._build_signal_metrics(w_trans, yaw_angle, point_idx, 'w_trans'),
                    'coord': coord_data,
                    'U_WT': u_wt_data,
                }

                measurement_point_key = f"{point_idx:02d}"
                self.processed_data[yaw_angle][measurement_point_key] = processed_metrics

        np.savez("processed_data.npz", processed_data=self.processed_data)
        self._print_quality_assessment()


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



# Execution
if __name__ == "__main__":

    file_paths = ["C:/dev/wind-tunnel-testing/measurements/yaw_minus30/Results_transient_17-6-2026_FRAP_group02_yawM30.mat", 
                  "C:/dev/wind-tunnel-testing/measurements/yaw_00/Results_transient_17-6-2026_FRAP_group02_yaw00.mat",
                  "C:/dev/wind-tunnel-testing/measurements/yaw_plus30/Results_transient_17-6-2026_FRAP_group02_yawP30.mat"]
    
    mat_reader = MatFilereader(file_paths)
    data_processor = DataProcessor(mat_reader.rawdata)

    # Initialize generator with explicit parameters, omitting fallbacks
    wake_map_generator = WakeMapGenerator(data_processor.processed_data, "wake_maps", 1.0)
    
    # Show all plots in the end
    plt.show()