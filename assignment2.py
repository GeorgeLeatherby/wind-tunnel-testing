import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def assignment2(file_path):
    # 1. Load Data
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    reynolds_struct = mat_data["Reynolds"]
    
    # Use index 3 for the high-Re case (Re = 90,000)
    data = reynolds_struct.LookUp[3]
    r_m = data.rm
    cord = data.cord

    # --- REYNOLDS CALCULATION AT MID-SPAN (Task 1) ---
    D = 1.1                 # Diameter [m]
    R = D / 2               # Radius [m]
    U_r = 5.7               # Rated wind speed [m/s]
    omega_rpm = 850         # Rated rotational speed [RPM]
    a = 1/3                 # Optimal axial induction
    nu = 1.511e-5           # Kinematic viscosity of air
    
    r_target = R / 2 
    omega_rads = omega_rpm * (2 * np.pi / 60)
    c_mid = np.interp(r_target, r_m, cord)
    
    U_axial = U_r * (1 - a)
    V_tangential = omega_rads * r_target
    V_rel = np.sqrt(U_axial**2 + V_tangential**2)
    Re_calculated = (V_rel * c_mid) / nu

    # --- DATA RETRIEVAL ---
    cp = data.Cp
    ct = data.Cf             # Thrust coefficient is 'Cf' in the dataset
    pitch = data.Pitch       # X-axis (151 elements)
    tsr = data.TSR           # Y-axis (60 elements)

    # Task 2: Find Optimal Point
    max_idx = np.unravel_index(np.nanargmax(cp), cp.shape) # (TSR_idx, Pitch_idx)
    opt_tsr = tsr[max_idx[0]]
    opt_pitch = pitch[max_idx[1]]
    max_cp = cp[max_idx]

    # --- TASK 3: SENSITIVITY ANALYSIS ---
    # Effect of Pitching (holding TSR constant at optimal)
    cp_vs_pitch = cp[max_idx[0], :]
    ct_vs_pitch = ct[max_idx[0], :]

    # Effect of TSR / Rotor Speed (holding Pitch constant at optimal)
    cp_vs_tsr = cp[:, max_idx[1]]
    ct_vs_tsr = ct[:, max_idx[1]]

    # --- CONSOLE OUTPUT ---
    print("-" * 30)
    print(f"TASK 1: REYNOLDS AT MID-SPAN: {Re_calculated:.0f}")
    print(f"TASK 2: OPTIMAL Cp: {max_cp:.4f} at TSR: {opt_tsr:.2f}, Pitch: {opt_pitch:.2f}°")
    print("-" * 30)

    # --- WINDOW 1: Geometry Plot ---
    r_nondim = (r_m - np.min(r_m)) / (np.max(r_m) - np.min(r_m))
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig1.canvas.manager.set_window_title('Task 1: Blade Geometry')
    ax1.plot(r_m, cord, 'o-'); ax1.set_title('Chord vs Radius'); ax1.grid(True, alpha=0.3)
    ax2.plot(r_nondim, cord, 'o-', color='orange'); ax2.set_title('Normalized Radius'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- WINDOW 2: Efficiency (Cp) Contour ---
    plt.figure(figsize=(8, 6))
    plt.gcf().canvas.manager.set_window_title('Task 2: Cp Contour')
    plt.contourf(pitch, tsr, cp, levels=20, cmap='viridis')
    plt.colorbar(label='Power Coefficient $C_p$')
    plt.plot(opt_pitch, opt_tsr, 'r*', markersize=12, label='Optimum')
    plt.title('Power Coefficient ($C_p$) Map')
    plt.xlabel('Pitch [°]'); plt.ylabel('TSR [-]'); plt.legend()
    plt.xlim(-10, 20)
    # --- WINDOW 3: Sensitivity Analysis (Task 3) ---
    fig3, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig3.canvas.manager.set_window_title('Task 3: Sensitivity Analysis')


    # Row 1: Cp Sensitivity
    axs[0, 0].plot(pitch, cp_vs_pitch, color='blue')
    axs[0, 0].set_title(f'$C_p$ vs Pitch (at TSR={opt_tsr:.1f})')
    axs[0, 0].set_ylabel('$C_p$')

    axs[0, 1].plot(tsr, cp_vs_tsr, color='blue')
    axs[0, 1].set_title(f'$C_p$ vs TSR (at Pitch={opt_pitch:.1f}°)')

    # Row 2: Ct Sensitivity
    axs[1, 0].plot(pitch, ct_vs_pitch, color='red')
    axs[1, 0].set_title(f'$C_T$ vs Pitch (at TSR={opt_tsr:.1f})')
    axs[1, 0].set_ylabel('$C_T$')
    axs[1, 0].set_xlabel('Pitch [°]')

    axs[1, 1].plot(tsr, ct_vs_tsr, color='red')
    axs[1, 1].set_title(f'$C_T$ vs TSR (at Pitch={opt_pitch:.1f}°)')
    axs[1, 1].set_xlabel('TSR [-]')

    for ax in axs.flat: ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # --- WINDOW 4: Thrust (Ct) Contour ---
    plt.figure(figsize=(8, 6))
    plt.gcf().canvas.manager.set_window_title('Task 3: Ct Contour')
    plt.contourf(pitch, tsr, ct, levels=20, cmap='magma')
    plt.colorbar(label='Thrust Coefficient $C_T$')
    plt.title('Thrust Coefficient ($C_T$) Map')
    plt.xlabel('Pitch [°]'); plt.ylabel('TSR [-]')

    plt.show()

# Execute
assignment2('LookUpTable_G1.mat')