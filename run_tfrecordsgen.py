from src import utils
import numpy as np
import concurrent.futures
import time
import argparse
import os
from npy_append_array import NpyAppendArray
import logging
from tqdm import tqdm  # <-- added

# --- Logger for utils.py ---
utils_logger = logging.getLogger('src.utils')
utils_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('utils_debug.log', mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
utils_logger.addHandler(file_handler)
utils_logger.propagate = False

# --- Logger for the main script ---
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def read_txt_to_list(file_path, input_dir, add_prefix=True):
    lines = []
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist."

    # Count total lines for tqdm
    with open(file_path, 'r', encoding='utf-8') as file:
        all_lines = file.readlines()

    for line in tqdm(all_lines, desc="Reading PDB IDs", unit="ID"):
        a = line.strip()
        if add_prefix:
            variants = [
                f'AF-{a}-F1-model_v4.pdb',
                f'AF-{a}-F1-model_v3.pdb',
                f'AF-{a}-F1-model_v2.pdb',
                f'AF-{a}-F1-model_v1.pdb'
            ]
        else:
            variants = [a, f'{a}.pdb']

        for pdb_name in variants:
            path = os.path.join(input_dir, pdb_name)
            if os.path.exists(path):
                lines.append(path)
                break
            else:
                print(f"Warning: {pdb_name} not found in {input_dir}. Skipping.")
    return lines


def process_pdb(pdb):
    path = utils.Extract_and_Save_from_PDB(
        input_file=pdb,
        from_dill=False,
        saving_dir='tmp/',
        k_nearest=5,
        inteacting_residues=False,
        un_dn=False,
        outtype='tfrecord',
        save_file=True,
        check_if_exists=True
    )
    
    
    return path


def main():
    parser = argparse.ArgumentParser(description="Process PDB files and generate TFRecords.")

    parser.add_argument('--pdb_ids_txt', type=str, help='Path to the text file containing PDB IDs. One per line.', required=True)
    parser.add_argument('--tfrecord_path', type=str, help='Path to save the TFRecord file.', required=True)
    parser.add_argument('--input_dir', type=str, default='/scratch-scc/users/u14286/piplines/Bind/data/large/alphafold_db/pdb')
    parser.add_argument('--tmp_dir', type=str, default='tmp', help='Temporary directory for intermediate files.')

    args = parser.parse_args()

    pdbs = read_txt_to_list(args.pdb_ids_txt, args.input_dir, add_prefix=True)
    print(f"Found {len(pdbs)} PDB files to process.")

    s = time.time()
    ### === Using ProcessPoolExecutor for parallel processing === ###
    with concurrent.futures.ProcessPoolExecutor() as executor:
        npz_paths = [executor.submit(process_pdb, pdb) for pdb in pdbs]
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(npz_paths), total=len(npz_paths), desc="Processing PDBs"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except:
                pass


    FEATURES, LABELS, IDs, PLDDT = [], [], [], []
    for path in tqdm(results, desc="Creating Arrays"):
        data = np.load(path)
    # Access each array by its key
        features = data['features']
        labels_arr = data['labels']
        ids = data['IDs']
        plddt = data['plddt']
        LABELS.append(labels_arr)
        IDs.append(ids)
        PLDDT.append(plddt)
        FEATURES.append(features)

    FEATURES = np.concatenate(FEATURES, axis=0)
    LABELS = np.concatenate(LABELS, axis=0)
    IDs = np.concatenate(IDs, axis=0)
    PLDDT = np.concatenate(PLDDT, axis=0)

    tfmanager = utils.TFRecordManager(tfrecord_path=args.tfrecord_path, feature_dim=301, plddt=True)
    tfmanager.write_samples(features=FEATURES, labels=LABELS, ID_arrs=IDs, plddt=PLDDT)

    train_ds = tfmanager.read_dataset()
    print("Finished processing in parallel:", time.time() - s)
    for batch in train_ds:
        features, labels, ID_arrs, plddt = batch
        print(features.shape, labels.shape, ID_arrs.shape, plddt.shape if plddt is not None else "No plddt")
        break


if __name__ == "__main__":
    main()
