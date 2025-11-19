"""
Processes a sequence of .fld files to generate augmented fields
based on the pysemtools example '6-Stats_from_fld.ipynb'.
"""

# Import required modules
import os
import numpy as np
from mpi4py import MPI

# Import pysemtools modules
try:
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.coef import Coef
    from pysemtools.io.ppymech.neksuite import pynekread, pynekwrite
    from pysemtools.postprocessing.statistics.fld_stats import generate_augmented_field
except ImportError:
    print("Error: pysemtools or its dependencies not found.")
    print("Please ensure pysemtools is installed correctly in your environment.")
    MPI.COMM_WORLD.Abort()

# --- User Configuration ---

# Path to your data files
FOLDER_PATH = "./"

# Path to save the new augmented files
OUTPUT_PATH = "./"

# File naming
FILE_PREFIX = "field0"
OUTPUT_PREFIX = "augmented_field0"

# Range of files to process
# Your files are field0.f00101 to field0.f00200
START_INDEX = 101
END_INDEX = 200

# Backend for calculations (as seen in the example)
BACKEND = "numpy"
DATA_DTYPE = np.float32
WRITE_WORD_SIZE = 4

# --- End Configuration ---

def main():
    # Get MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # 1. Create the file sequence
    file_indices = range(START_INDEX, END_INDEX + 1)
    file_sequence = [
        os.path.join(FOLDER_PATH, f"{FILE_PREFIX}.f{str(i).zfill(5)}")
        for i in file_indices
    ]

    if not file_sequence:
        if rank == 0:
            print(f"Error: No files found for prefix '{FILE_PREFIX}' in range {START_INDEX}-{END_INDEX}.")
        return

    # 2. Read the mesh and create coefficients
    if rank == 0:
        print(f"Reading mesh from: {file_sequence[0]}")

    # Initialize empty Mesh object
    msh = Mesh(comm=comm, bckend=BACKEND)
    
    # Read mesh data from the first file
    pynekread(comm=comm, filename=file_sequence[0], msh=msh, data_dtype=DATA_DTYPE)

    # Generate the coefficients (for derivatives) from the mesh
    coef = Coef(msh=msh, comm=comm, bckend=BACKEND)

    if rank == 0:
        print("Mesh and coefficients initialized.")
        print("Starting file processing loop...")

    # 3. Loop over all files, process, and write
    for i, fname in enumerate(file_sequence):
        if rank == 0:
            print("===============================================")
            print(f"Processing file: {fname}")
        
        # Get the original file number (e.g., 101)
        file_number = START_INDEX + i

        # Initialize an empty FieldRegistry
        fld = FieldRegistry(comm=comm, bckend=BACKEND)
        
        # Read the field data from the current file
        pynekread(comm=comm, filename=fname, fld=fld, data_dtype=DATA_DTYPE)

        # Call the routine to generate augmented fields
        augmented_fld = generate_augmented_field(
            comm=comm, msh=msh, fld=fld, coef=coef, dtype=msh.x.dtype
        )

        # Define the output filename
        output_filename = os.path.join(
            OUTPUT_PATH, f"{OUTPUT_PREFIX}.f{str(file_number).zfill(5)}"
        )

        # Write the augmented data to the new file
        # Note: We convert back to numpy for writing, as shown in the example
        pynekwrite(
            comm=comm, 
            filename=output_filename, 
            msh=msh.to(comm=comm, bckend="numpy"), 
            fld=augmented_fld.to(comm=comm, bckend="numpy"), 
            wdsz=WRITE_WORD_SIZE
        )
        
        if rank == 0:
            print(f"Successfully wrote: {output_filename}")

    if rank == 0:
        print("===============================================")
        print("All files processed successfully.")

if __name__ == "__main__":
    main()