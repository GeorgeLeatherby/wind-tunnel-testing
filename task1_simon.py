"""
Preamble
"""

# Imports
import numpy as np
import scipy

class task1:
    """
    Task 1
    """

    def __init__(self):
        self.file_path = "LookUpTable_G1.mat"

        self.requested_reynolds = 75000
        self.tsr_array = np.linspace(6.0, 9.0, 30)  # (tip speed ratio) range to test
        self.yaw_angle = [0.0, 15.0, 30.0]          # (deg) yaw angle of rotor relative to incoming flow
        self.pitch_angle = [0.0]                    # (deg) pitch angle of rotor blades

        self.run()

    def run(self):
        """
        Run the task
        """
        print("Running Task 1")
        self.load_matlab_data(self.file_path)


    def load_matlab_data(self, file_path):
        """
        Load the matlab data
        """
        print("Loading Matlab data")
        self.df = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        self.reynolds_struct = self.df["Reynolds"]

        available_reynolds = [self.reynolds_struct.LookUp[i].Re for i in range(len(self.reynolds_struct.LookUp))]

        # find index of the requested reynolds number
        reynolds_idx = np.argmin(np.abs(np.array(available_reynolds) - self.requested_reynolds))

        if available_reynolds[reynolds_idx] != self.requested_reynolds:
            print(f"Warning: Requested Reynolds number {self.requested_reynolds} not found. Using closest available Reynolds number {available_reynolds[reynolds_idx]} instead.")
    
        # Use index 2 for the required Re case (Re = 75,000)
        self.data = self.reynolds_struct.LookUp[reynolds_idx]

        # --- DATA RETRIEVAL ---
        self.r_m = self.data.rm         # Radial positions along the blade span (m)
        self.cord = self.data.cord      # Chord length distribution along the blade span
        self.cp = self.data.Cp          # Power coefficient is 'Cp' in the dataset
        self.ct = self.data.Cf          # Thrust coefficient is 'Cf' in the dataset
        self.pitch = self.data.Pitch    # X-axis (151 elements)
        self.tsr = self.data.TSR        # Y-axis (60 elements)

    def calculate_u_infinity_dash(self):
        """
        Calculate the wind speed to request from wind tunnel
        """


# Execution
task1 = task1()