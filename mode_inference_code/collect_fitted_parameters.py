import os
import glob
import pandas as pd
import numpy as np

# Set the directory path where the csv files are stored
directory_path = "./allIRs_20modes_v6/"

# Get all the directories in the directory_path and save them as a csv
directories = glob.glob(os.path.join(directory_path, "*"))
directories = [os.path.basename(os.path.normpath(d)) for d in directories]
# Check if each element in the list is a directory
directories = [d for d in directories if os.path.isdir(os.path.join(directory_path, d))]
directories = pd.DataFrame(directories)
directories.to_csv(directory_path+"/directories.csv", index=False)
print(directories[0][0])

csv_files_RT60 = [None] * len(directories)
csv_files_Amp = [None] * len(directories)
csv_files_freq = [None] * len(directories)
csv_files_alpha = [None] * len(directories)
csv_files_gamma = [None] * len(directories)

    
    
csv_files_RT60 = glob.glob(os.path.join(directory_path, "**", "RT60.csv"), recursive=True)
csv_files_Amp = glob.glob(os.path.join(directory_path, "**", "Amp.csv"), recursive=True)
csv_files_freq = glob.glob(os.path.join(directory_path, "**", "freq.csv"), recursive=True)
csv_files_alpha = glob.glob(os.path.join(directory_path, "**", "alpha.csv"), recursive=True)
csv_files_gamma = glob.glob(os.path.join(directory_path, "**", "gamma.csv"), recursive=True)

# put oberserved_1.wav from each folder into a separate directory called observed by renaming them according to their containing subdirectories
for ind in directories.index:
    directory_path2 = directory_path+str(directories[0][ind]) +"/"
    # look for observed_1.wav in directory_path2
    observed_files = glob.glob(os.path.join(directory_path2, "observed_1.wav"))
    # copy the observed_1.wav to a new directory called observed

# size of the csv_files_Amp

# Create an empty list to store the dataframes
np_array1 = []
np_array2 = []
np_array3 = []
np_array4 = []
np_array5 = []

# Loop through each each of the scraped csv files and store as numpy arrays
for indd in range(len(csv_files_RT60)):
    df = pd.read_csv(csv_files_RT60[indd], header=None)
    np_array1.append(df.to_numpy())
    df = pd.read_csv(csv_files_Amp[indd], header=None)
    np_array2.append(df.to_numpy())
    df = pd.read_csv(csv_files_freq[indd], header=None)
    np_array3.append(df.to_numpy())
    df = pd.read_csv(csv_files_alpha[indd], header=None)
    np_array4.append(df.to_numpy())
    df = pd.read_csv(csv_files_gamma[indd], header=None)
    np_array5.append(df.to_numpy())




# Concatenate each of the np_arrays into 2D matrices
np_array1 = np.concatenate(np_array1, axis=0)
np_array2 = np.concatenate(np_array2, axis=0)
np_array3 = np.concatenate(np_array3, axis=0)
np_array4 = np.concatenate(np_array4, axis=0)
np_array5 = np.concatenate(np_array5, axis=0)


#reshape np_array into 2D matrix with the 0 index being of length 10

np_array1 = np_array1.reshape(-1,20)
np_array2 = np_array2.reshape(-1,20)
np_array3 = np_array3.reshape(-1,20)
np_array4 = np_array4.reshape(-1,20)
np_array5 = np_array5.reshape(-1,20)

# Write the combined dataframe to a new csv file
np.savetxt(directory_path+"/RT60_1.csv", np_array1, delimiter=",")
np.savetxt(directory_path+"/Amp_1.csv", np_array2, delimiter=",")
np.savetxt(directory_path+"/freq_1.csv", np_array3, delimiter=",")
np.savetxt(directory_path+"/alpha_1.csv", np_array4, delimiter=",")
np.savetxt(directory_path+"/gamma_1.csv", np_array5, delimiter=",")
np.savetxt(directory_path+"/directories.csv", csv_files_RT60, delimiter=",", fmt='%s')
