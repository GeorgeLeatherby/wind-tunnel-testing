import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io, stats, signal, integrate
import os

class DataLoader:
    """
    Handles the ingestion and formatting of .mat files containing wind tunnel probe data.
    Follows the F.I.R.S.T principle by being Independent and Self-Checking.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data = {}
        
    def load_all(self):
        for dist, path in self.file_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file {path} not found in the working directory.")
            
            mat = io.loadmat(path, squeeze_me=True, struct_as_record=False)
            wake_data = mat['Wake']
            
            probes = [name for name in wake_data._fieldnames if name.startswith('Probe')]
            
            extracted_probes = []
            for probe_name in probes:
                probe = getattr(wake_data, probe_name)

                # Identify indices where ANY velocity component contains a NaN value
                ux_raw = np.asarray(probe.Ux)
                uy_raw = np.asarray(probe.Uy)
                uz_raw = np.asarray(probe.Uz)
                valid_mask = ~(np.isnan(ux_raw) | np.isnan(uy_raw) | np.isnan(uz_raw))
                 
                # Helper function to safely mask arrays while leaving scalar positions/constants untouched
                def clean_attr(val):
                    arr = np.asarray(val)
                    return arr[valid_mask] if arr.ndim > 0 and arr.shape == valid_mask.shape else val
                 
                extracted_probes.append({
                        'name': probe_name,
                        'X': clean_attr(probe.Position.X),
                        'Y': clean_attr(probe.Position.Y),
                        'Z': clean_attr(probe.Position.Z),
                        'Ux': clean_attr(probe.Ux),
                        'Uy': clean_attr(probe.Uy),
                        'Uz': clean_attr(probe.Uz),
                        'Rho': clean_attr(getattr(probe, 'Rho', 1.225))
                })

            
            # Sort probes by Y coordinate horizontally
            df = pd.DataFrame(extracted_probes).sort_values(by='Y').reset_index(drop=True)
            self.data[dist] = df
        return self.data


class TurbulenceAnalyzer:
    """
    Handles all statistical and turbulence-specific calculations.
    """
    def __init__(self, fs=2000):
        self.fs = fs

    def compute_profile_statistics(self, df):
        y_pos = df['Y'].values
        mean_u = np.array([np.mean(u) for u in df['Ux']])
        mean_v = np.array([np.mean(v) for v in df['Uy']])
        mean_w = np.array([np.mean(w) for w in df['Uz']])
        
        std_u = np.array([np.std(u) for u in df['Ux']])
        std_v = np.array([np.std(v) for v in df['Uy']])
        std_w = np.array([np.std(w) for w in df['Uz']])
        
        # Turbulence Intensity (Local)
        ti_u = std_u / mean_u
        
        # Reynolds Stresses
        uu = np.array([np.mean((row['Ux'] - m_u)**2) for row, m_u in zip(df.to_dict('records'), mean_u)])
        vv = np.array([np.mean((row['Uy'] - m_v)**2) for row, m_v in zip(df.to_dict('records'), mean_v)])
        ww = np.array([np.mean((row['Uz'] - m_w)**2) for row, m_w in zip(df.to_dict('records'), mean_w)])
        uv = np.array([np.mean((row['Ux'] - m_u)*(row['Uy'] - m_v)) for row, m_u, m_v in zip(df.to_dict('records'), mean_u, mean_v)])
        uw = np.array([np.mean((row['Ux'] - m_u)*(row['Uz'] - m_w)) for row, m_u, m_w in zip(df.to_dict('records'), mean_u, mean_w)])
        vw = np.array([np.mean((row['Uy'] - m_v)*(row['Uz'] - m_w)) for row, m_v, m_w in zip(df.to_dict('records'), mean_v, mean_w)])
        
        return y_pos, mean_u, std_u, ti_u, uu, vv, ww, uv, uw, vw

    def compute_uncertainty(self, mean_u, std_u, n_samples):
        # 99.9% confidence interval z-score is approx 3.29
        z_score = 3.29 
        random_error = z_score * (std_u / np.sqrt(n_samples))
        
        # Systematic error polynomial from calibration curve
        systematic_error = 0.007458 * (mean_u**2) - 0.1186 * mean_u + 0.5876
        
        total_uncertainty = np.sqrt(random_error**2 + systematic_error**2)
        return total_uncertainty

    def calculate_macro_scales(self, u_signal):
        u_mean = np.mean(u_signal)
        u_fluct = u_signal - u_mean
        variance = np.var(u_fluct)
        
        # Autocorrelation
        autocorr = signal.correlate(u_fluct, u_fluct, mode='full')
        autocorr = autocorr[len(autocorr)//2:] / (variance * len(u_fluct))
        
        # Find first zero crossing for integration limit
        zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
        if len(zero_crossings) > 0:
            limit = zero_crossings[0]
        else:
            limit = len(autocorr)
            
        time_lags = np.arange(limit) / self.fs
        
        # Integrate using Trapezoidal rule
        macro_time_scale = integrate.trapezoid(autocorr[:limit], time_lags)
        macro_length_scale = u_mean * macro_time_scale # Taylor's frozen turbulence
        
        return time_lags, autocorr[:limit], macro_time_scale, macro_length_scale


class AerodynamicModels:
    """
    Handles domain-specific physics equations like Wake Models and Power Availability.
    """
    def __init__(self, u_inf=5.45, R_mm=550.0, kw=0.075, Ct=8/9, rho=1.156):
        self.u_inf = u_inf
        self.R = R_mm / 1000.0  # convert to meters
        self.kw = kw
        self.Ct = Ct
        self.rho = rho
        
    def jensen_wake(self, x_D, y_mm):
        # x is distance in terms of D. We need it in meters. D = 2*R
        x_m = x_D * (2 * self.R)
        y_m = y_mm / 1000.0
        
        wake_radius = self.R + self.kw * x_m
        deficit = (1 - np.sqrt(1 - self.Ct)) / ((1 + (self.kw * x_m) / self.R)**2)
        
        u_wake = np.ones_like(y_m) * self.u_inf
        in_wake = np.abs(y_m) <= wake_radius
        u_wake[in_wake] = self.u_inf * (1 - deficit)
        return u_wake
        
    def compute_available_power(self, y_pos_mm, mean_u, Cp=0.593):
        # Filter profile to rotor bounds [-R, R]
        mask = (y_pos_mm >= -self.R * 1000) & (y_pos_mm <= self.R * 1000)
        u_disk = mean_u[mask]
        
        # Area = pi * R^2
        area = np.pi * (self.R ** 2)
        
        # Rotor averaged velocity (approximated as 1D mean over disk span)
        u_avg = np.mean(u_disk)
        p_avail = 0.5 * self.rho * area * Cp * (u_avg ** 3)
        
        return p_avail


class WindTunnelPlotter:
    """
    Visualizes the processed data according to tasks 1-8.
    """
    def __init__(self, data, analyzer, aero_model):
        self.data = data
        self.analyzer = analyzer
        self.aero_model = aero_model

    def run_all_tasks(self):
        self.task_1()
        self.task_2_and_8()
        self.task_3()
        self.task_4()
        self.task_5()
        self.task_6()
        self.task_7()
        plt.show()

    def task_1(self):
        df_5D = self.data['5D']
        y_pos, m_u, s_u, ti_u, uu, vv, ww, uv, uw, vw = self.analyzer.compute_profile_statistics(df_5D)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Task 1: Horizontal Wake Profiles at 5D')
        
        axs[0, 0].plot(y_pos, m_u, 'b-o')
        axs[0, 0].set_title('Mean Velocity Ux')
        axs[0, 0].set_ylabel('Velocity [m/s]')
        
        axs[0, 1].plot(y_pos, s_u, 'r-o')
        axs[0, 1].set_title('Standard Deviation of Ux')
        
        axs[1, 0].plot(y_pos, ti_u * 100, 'g-o')
        axs[1, 0].set_title('Turbulence Intensity (%)')
        axs[1, 0].set_ylabel('TI [%]')
        
        axs[1, 1].plot(y_pos, uu, label="u'u'")
        axs[1, 1].plot(y_pos, vv, label="v'v'")
        axs[1, 1].plot(y_pos, ww, label="w'w'")
        axs[1, 1].plot(y_pos, uv, label="u'v'")
        axs[1, 1].set_title('Reynolds Stresses')
        axs[1, 1].legend()
        
        for ax in axs.flat:
            ax.set_xlabel('y position [mm]')
            ax.grid(True)
        plt.tight_layout()

    def task_2_and_8(self):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Task 2 & 8: Mean Velocity Profiles and TI with Uncertainty')
        
        for dist in ['5D', '7.5D', '10D']:
            df = self.data[dist]
            y_pos, m_u, s_u, ti_u, _, _, _, _, _, _ = self.analyzer.compute_profile_statistics(df)
            n_samples = len(df['Ux'][0])
            
            # Task 8: Uncertainty calculation
            uncertainty = self.analyzer.compute_uncertainty(m_u, s_u, n_samples)
            
            axs[0].errorbar(y_pos, m_u, yerr=uncertainty, fmt='-o', label=f'x = {dist}', capsize=3)
            axs[1].plot(y_pos, ti_u * 100, '-o', label=f'x = {dist}')
            
        axs[0].set_title('Mean Velocity (Ux) with 99.9% Conf. Error Bars')
        axs[0].set_ylabel('Velocity [m/s]')
        axs[1].set_title('Turbulence Intensity')
        axs[1].set_ylabel('TI [%]')
        
        for ax in axs:
            ax.set_xlabel('y position [mm]')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()

    def task_3(self):
        plt.figure(figsize=(10, 6))
        plt.title('Task 3: Jensen Wake Model vs Experimental Data')
        
        colors = ['b', 'r', 'g']
        for i, dist_str in enumerate(['5D', '7.5D', '10D']):
            x_D = float(dist_str.replace('D', ''))
            df = self.data[dist_str]
            y_pos, m_u, _, _, _, _, _, _, _, _ = self.analyzer.compute_profile_statistics(df)
            
            jensen_u = self.aero_model.jensen_wake(x_D, y_pos)
            
            plt.plot(y_pos, m_u, 'o', color=colors[i], label=f'Exp {dist_str}')
            plt.plot(y_pos, jensen_u, '-', color=colors[i], label=f'Jensen {dist_str}')
            
        plt.xlabel('y position [mm]')
        plt.ylabel('Velocity Ux [m/s]')
        plt.legend()
        plt.grid(True)

    def task_4(self):
        distances = ['5D', '7.5D', '10D']
        x_vals = [5, 7.5, 10]
        power_vals = []
        
        for dist in distances:
            df = self.data[dist]
            y_pos, m_u, _, _, _, _, _, _, _, _ = self.analyzer.compute_profile_statistics(df)
            p_avail = self.aero_model.compute_available_power(y_pos, m_u)
            power_vals.append(p_avail)
            
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, power_vals, 's-m', markersize=10)
        plt.title('Task 4: Available Power vs Downstream Distance')
        plt.xlabel('Downstream distance [x/D]')
        plt.ylabel('Available Power [W]')
        plt.grid(True)

    def task_5(self):
        # Pick the center probe at 5D for PSD
        df = self.data['5D']
        center_probe = df.iloc[len(df)//2]
        u_signal = center_probe['Ux']
        
        f, Pxx = signal.welch(u_signal, fs=self.analyzer.fs, nperseg=2048)
        
        plt.figure(figsize=(8, 6))
        plt.loglog(f, Pxx, label='Measured PSD')
        
        # Plot -5/3 slope reference line
        f_ref = f[(f > 10) & (f < 100)]
        if len(f_ref) > 0:
            p_ref = Pxx[(f > 10) & (f < 100)][0] * (f_ref / f_ref[0]) ** (-5/3)
            plt.loglog(f_ref, p_ref, 'k--', linewidth=2, label='-5/3 Inertial Subrange')
            
        plt.title('Task 5: Power Spectral Density of Velocity')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [(m/s)²/Hz]')
        plt.legend()
        plt.grid(True, which="both", ls="--")

    def task_6(self):
        df = self.data['5D']
        y_targets = [-1050, -50, 1050]
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Task 6: PDF and Skewness at Specific Y Locations')
        
        for i, target in enumerate(y_targets):
            # Find closest Y
            idx = (np.abs(df['Y'] - target)).idxmin()
            probe = df.iloc[idx]
            u_signal = probe['Ux']
            
            skewness = stats.skew(u_signal)
            mean, std = np.mean(u_signal), np.std(u_signal)
            
            axs[i].hist(u_signal, bins=50, density=True, alpha=0.6, color='b', label='Histogram')
            
            # Gaussian fit
            xmin, xmax = axs[i].get_xlim()
            x_fit = np.linspace(xmin, xmax, 100)
            p_fit = stats.norm.pdf(x_fit, mean, std)
            axs[i].plot(x_fit, p_fit, 'k', linewidth=2, label='Gaussian Fit')
            
            axs[i].set_title(f'Y $\\approx$ {probe["Y"]} mm\nSkewness = {skewness:.3f}')
            axs[i].set_xlabel('Velocity Ux [m/s]')
            axs[i].legend()
            
        plt.tight_layout()

    def task_7(self):
        # Pick center probe at 5D
        df = self.data['5D']
        center_probe = df.iloc[len(df)//2]
        u_signal = center_probe['Ux']
        
        time_lags, autocorr, t_macro, l_macro = self.analyzer.calculate_macro_scales(u_signal)
        
        plt.figure(figsize=(8, 5))
        plt.plot(time_lags, autocorr, 'b-')
        plt.title(f'Task 7: Auto-correlation at center wake (5D)\n$T_{{macro}}$ = {t_macro:.4f} s, $L_{{macro}}$ = {l_macro:.4f} m')
        plt.xlabel('Time lag $\\tau$ [s]')
        plt.ylabel('Autocorrelation $R(\\tau)$')
        plt.grid(True)

if __name__ == "__main__":
    file_mapping = {
        '5D': 'horizontal_5D_yaw0.mat',
        '7.5D': 'horizontal_75D_yaw0.mat',
        '10D': 'horizontal_10D_yaw0.mat'
    }
    
    try:
        # 1. Initialize and Load Data
        loader = DataLoader(file_mapping)
        data = loader.load_all()
        
        # 2. Instantiate physics and math tools
        analyzer = TurbulenceAnalyzer(fs=2000)
        aero_model = AerodynamicModels(u_inf=5.45, R_mm=550.0)
        
        # 3. Process and Plot
        plotter = WindTunnelPlotter(data, analyzer, aero_model)
        plotter.run_all_tasks()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset .mat files are in the same directory as this script.")