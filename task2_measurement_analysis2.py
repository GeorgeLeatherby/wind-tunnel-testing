import os
import mat73
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

class MatFileReader:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.rawdata = {}

        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
            mat_data = mat73.loadmat(file_path, use_attrdict=True)
            self.rawdata[file_path] = mat_data

        print("Raw measurement data successfully read from .mat files.")

class DataProcessor:
    def __init__(self, rawdata):
        self.rawdata = rawdata
        self.processed_data = {}
        self.sampling_frequency = 10000  # 10 kHz
        self.nperseg = 4096              # High resolution for 100k data points
        self.data_processing() 

    def data_processing(self):
        for file_path, mat_data in self.rawdata.items():
            # Standardize naming across systems
            yaw_clean = file_path.replace("\\", "/").split("/")[-1]
            if "yawM30" in yaw_clean: yaw_angle = "-30"
            elif "yaw00" in yaw_clean: yaw_angle = "0"
            elif "yawP30" in yaw_clean: yaw_angle = "+30"
            else: yaw_angle = "unknown"

            self.processed_data[yaw_angle] = {}
            resultscell = mat_data['resultscell']

            for point_index, measurement_point in enumerate(resultscell):
                measurement_point_key = f"{point_index:02d}"
                processed_metrics = {}

                # CRITICAL FIX: Ensure extraction targets a true 1D array profile
                # Check shape of element and flatten/extract the dominant time series axis
                def clean_1d_array(raw_element):
                    arr = np.array(raw_element)
                    if arr.ndim > 1:
                        # If shape is (100000, 3) or similar, take the longest dimension or column 0
                        if arr.shape[0] > arr.shape[1]:
                            arr = arr[:, 0]
                        else:
                            arr = arr[0, :]
                    return np.squeeze(arr)

                # Extract and verify 1D structure
                velocity_data = clean_1d_array(measurement_point[1][5]) # Index 5 maps to velocity
                
                # Extract coordinates from matrix payload safely
                raw_coords = np.squeeze(np.array(measurement_point[1][10]))
                # If coord vector has multiple elements, map Y and Z
                y_coord = raw_coords[0] if raw_coords.ndim > 0 and len(raw_coords) > 0 else point_index * 10.0
                z_coord = raw_coords[1] if raw_coords.ndim > 0 and len(raw_coords) > 1 else 0.0
                processed_metrics['coordinates'] = (y_coord, z_coord)

                # Compute standard statistics
                mean_val = np.mean(velocity_data)
                median_val = np.median(velocity_data)
                std_dev = np.std(velocity_data)

                # Spectral execution with explicit axis isolation
                f_welch, psd_welch = welch(velocity_data, fs=self.sampling_frequency, nperseg=self.nperseg, axis=-1)

                # Find peaks safely on verified 1D psd array
                peak_signals, properties = find_peaks(psd_welch, height=np.max(psd_welch) * 0.1)
                
                if len(peak_signals) > 0:
                    highest_peak_idx = peak_signals[np.argmax(psd_welch[peak_signals])]
                    dominant_freq = f_welch[highest_peak_idx]
                    dominant_psd_mag = psd_welch[highest_peak_idx]
                else:
                    dominant_freq = 0.0
                    dominant_psd_mag = 0.0

                processed_metrics['Velocity'] = {
                    'mean': mean_val,
                    'median': median_val,
                    'std_dev': std_dev,
                    'fwelch': f_welch,
                    'psdwelch': psd_welch,
                    'dominant_freq': dominant_freq,
                    'dominant_psd_mag': dominant_psd_mag
                }

                self.processed_data[yaw_angle][measurement_point_key] = processed_metrics

        print("Data processing completed successfully without vector geometry errors.")

class WakeMapGenerator:
    def __init__(self, processed_data):
        self.processed_data = processed_data
        self.feature_tables = {}

    def phase1_diagnostic_heatmap(self):
        """Generates a spatial-spectral heatmap across all 23 nodes."""
        for yaw, points in self.processed_data.items():
            psd_list = []
            sorted_keys = sorted(points.keys())
            freqs = points['00']['Velocity']['fwelch']
            
            for pt in sorted_keys:
                psd_list.append(points[pt]['Velocity']['psdwelch'])
            
            psd_matrix = np.array(psd_list)
            
            # Limit plot window to 0-500Hz where wind turbine blade-pass frequencies occur
            cutoff_idx = np.where(freqs <= 500)[0][-1]
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(np.log10(psd_matrix[:, :cutoff_idx] + 1e-10), 
                        cmap='inferno', 
                        xticklabels=False, # Vector is too large for individual ticks
                        yticklabels=sorted_keys)
            plt.title(f'Phase 1: Spectral Footprint Heatmap - Yaw {yaw}° (0 - 500 Hz)')
            plt.xlabel('Frequency Spectrum $\\rightarrow$ Increasing')
            plt.ylabel('Spatial Measurement Point')
            plt.tight_layout()
            plt.savefig(f'Phase1_Heatmap_Yaw_{yaw}.png', dpi=300)
            plt.close()

    def phase2_feature_table(self, psd_threshold=0.05):
        """Builds an engineering decision ledger to determine metrics mathematically."""
        for yaw, points in self.processed_data.items():
            rows = []
            for pt_key in sorted(points.keys()):
                v_data = points[pt_key]['Velocity']
                y, z = points[pt_key]['coordinates']
                
                # Signal Classification Logic Matrix
                if v_data['dominant_psd_mag'] > psd_threshold and v_data['dominant_freq'] > 1.0:
                    signal_type = "Periodic (Vortex/Blade)"
                    selected_value = v_data['mean'] # Standard approach for periodic cyclic domains
                else:
                    signal_type = "Stochastic (Turbulent)"
                    selected_value = v_data['median'] # Robust choice against random transient bursts
                
                rows.append({
                    'Point_ID': pt_key, 'Y_mm': y, 'Z_mm': z,
                    'Mean_m_s': v_data['mean'], 'Median_m_s': v_data['median'],
                    'StdDev': v_data['std_dev'], 'Dom_Freq_Hz': v_data['dominant_freq'],
                    'Peak_PSD': v_data['dominant_psd_mag'], 'Signal_Type': signal_type,
                    'Selected_Velocity': selected_value
                })
            
            df = pd.DataFrame(rows)
            self.feature_tables[yaw] = df
            df.to_csv(f'Phase2_DecisionTable_Yaw_{yaw}.csv', index=False)

    def phase3_wake_map(self):
        """Generates continuous 2D wake contours based on spatial node matrix inputs."""
        for yaw, df in self.feature_tables.items():
            plt.figure(figsize=(9, 6))
            
            # Continuous interpolation using triangulated contour plotting
            contour = plt.tricontourf(df['Y_mm'], df['Z_mm'], df['Selected_Velocity'], levels=20, cmap='vlag')
            cbar = plt.colorbar(contour)
            cbar.set_label('Representative Target Velocity ($u$) [m/s]')
            
            # Map physical positions of nodes
            plt.scatter(df['Y_mm'], df['Z_mm'], color='black', marker='x', s=30, alpha=0.7, label='Sensor Nodes')
            plt.title(f'Phase 3: 2D Aerodynamic Wake Profile Map (Yaw = {yaw}°)')
            plt.xlabel('Cross-stream Axis Y [mm]')
            plt.ylabel('Vertical Axis Z [mm]')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f'Phase3_WakeMap_Yaw_{yaw}.png', dpi=300)
            plt.close()

if __name__ == "__main__":
    base_dir = "C:/dev/wind-tunnel-testing/measurements"
    file_paths = [
        f"{base_dir}/yaw_minus30/00_Results/Results_transient_17-6-2026_FRAP_group02_yawM30.mat",
        f"{base_dir}/yaw00/00_Results/Results_transient_17-6-2026_FRAP_group02_yaw00.mat",
        f"{base_dir}/yaw_plus30/00_Results/Results_transient_17-6-2026_FRAP_group02_yawP30.mat"
    ]
    
    mat_reader = MatFileReader(file_paths)
    if mat_reader.rawdata:
        processor = DataProcessor(mat_reader.rawdata)
        generator = WakeMapGenerator(processor.processed_data)
        
        print("Executing Pipeline Phases...")
        generator.phase1_diagnostic_heatmap()
        generator.phase2_feature_table(psd_threshold=0.01) # Tuning threshold based on power scales
        generator.phase3_wake_map()
        print("All visualization assets and feature arrays saved to working tree.")