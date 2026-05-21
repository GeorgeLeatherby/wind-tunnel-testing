Task: Make a Plan to achieve the requested goal. Review your plan before any execution. Carefully review that you understood all technical and physical requirements, as well as used the provided formulas from the files. Execute what is necessary to achieve the goal based on reviewed plan. Explain what you did and why this works. State all your assumptions.

Goal: Perform all tasks from Excercise 4in the way that was defined by the user prompt. Use the formulas provided by the chair (uploaded files). 

Deliverable: Python script in OOP form which can be used to calcualte all requested values from Excercise 4 (Tasks 1 - 8). 

Coding rules: Prefer simple implementation and proper definition and seperation of functions and classes. Follow the F.I.R.S.T. principle. Use commentary to describe the intended function of each function and class. Also explain scientific concepts used in commentary at the defined functions. Assume all mentioned files from the assignment to be available in the project folder (e.g. horizontal_5D_yaw0.mat). Prefer the use of the following packages: scipy, numpy, pandas, matplotlib.pyplot. Cleanly define all used variables in the constructors of classes for better overview. Do not introduce fallbacks or synthetic data! If a file is not found simply throw a fitting error.

File specifications:
Matlab file loaded using scipy.io.loadmat with squeeze_me=True and struct_as_record=False
Keys raw matlab file:
dict_keys(['__header__', '__version__', '__globals__', 'Wake'])
Extracted data type of 'Wake':
<class 'scipy.io.matlab._mio5_params.mat_struct'>
Field names of 'Wake': (extracted_data._fieldnames)
['Probe1', 'Probe2', 'Probe3', 'Probe4', 'Probe5', 'Probe6', 'Probe7', 'Probe8', 'Probe9', 'Probe10', 'Probe11', 'Probe12', 'Probe13', 'Probe14', 'Probe15', 'Probe16', 'Probe17', 'Probe18', 'Probe19', 'Probe20', 'Probe21', 'Probe22', 'Probe23', 'Probe24']
Probe1 fieldnames: (extracted_data.Probe1._fieldnames)
['InitTime', 'Rho', 'Pitot', 'Position', 'Ux', 'Uy', 'Uz']
Data of Probe1 in order:
5
1.156
5.437
<scipy.io.matlab._mio5_params.mat_struct object at 0x000002B2F1E64830>
[4.65774 4.65508 4.66522 ... 3.62617 3.63361 3.63365]
[-0.53166 -0.37106 -0.35763 ...  0.45156  0.43977  0.48389]
[ 0.2023   0.27246  0.32815 ... -0.23828 -0.23247 -0.18967]
Fieldnames of 'Position' in Probe1: (extracted_data.Probe1.Position._fieldnames)
['X', 'Y', 'Z']
Example data of field X in Position of Probe1: (extracted_data.Probe1.Position.X)
5500