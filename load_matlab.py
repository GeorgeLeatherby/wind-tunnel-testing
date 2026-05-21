import scipy.io as sio

filepath = "horizontal_5D_yaw0.mat"
mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)

print("Matlab file loaded using scipy.io.loadmat with squeeze_me=True and struct_as_record=False")
print("Keys raw matlab file:")
print(mat.keys())


extracted_data = mat['Wake']

# get structure and keys of array
print("Extracted data type of 'Wake':")
print(type(extracted_data))
print("Field names of 'Wake': (extracted_data._fieldnames)")
print(extracted_data._fieldnames)

print("Probe1 fieldnames: (extracted_data.Probe1._fieldnames)")
print(extracted_data.Probe1._fieldnames)

print("Data of Probe1 in order:")
print(extracted_data.Probe1.InitTime)
print(extracted_data.Probe1.Rho) 
print(extracted_data.Probe1.Pitot)
print(extracted_data.Probe1.Position)
print(extracted_data.Probe1.Ux)
print(extracted_data.Probe1.Uy)
print(extracted_data.Probe1.Uz)

print("Fieldnames of 'Position' in Probe1: (extracted_data.Probe1.Position._fieldnames)")
print(extracted_data.Probe1.Position._fieldnames)

print("Data of field X in Position of Probe1: (extracted_data.Probe1.Position.X)")
print(extracted_data.Probe1.Position.X)