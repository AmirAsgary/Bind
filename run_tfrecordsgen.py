from src import utils
import numpy as np
import concurrent.futures
import time
import argparse
import os
from npy_append_array import NpyAppendArray
import logging
from tqdm import tqdm 
import pandas as pd
from tqdm import tqdm  # <-- added
import pandas as pd

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

def tsv_to_list(file_path):
    df = pd.read_csv(file_path, sep='\t')
    dftrain = df[df['train1_valid2_test3'] == 1]
    for i in range(min(df.group), max(df.group) + 1):
        dftrain_grou = dftrain[dftrain['group']==i]
        repids = '\n'.join(dftrain_grou.repId.tolist()) # each id in one row
        with open(f'data/large/alphafold_db/train_test_val/train_group{i}.txt', 'w', encoding="utf-8") as f:
            f.write(repids)
    dfval = df[df['train1_valid2_test3'] == 2]
    dftest = df[df['train1_valid2_test3'] == 3]
    repids_val = '\n'.join(dfval.repId.tolist())
    repids_test = '\n'.join(dftest.repId.tolist())
    with open(f'data/large/alphafold_db/train_test_val/valid_all.txt', 'w', encoding="utf-8") as f:
        f.write(repids_val)
    with open(f'data/large/alphafold_db/train_test_val/test_all.txt', 'w', encoding="utf-8") as f:
        f.write(repids_test)

def tsv_to_list(file_path):
    df = pd.read_csv(file_path, sep='\t')
    dftrain = df[df['train1_valid2_test3'] == 1]
    for i in range(min(df.group), max(df.group) + 1):
        dftrain_grou = dftrain[dftrain['group']==i]
        repids = '\n'.join(dftrain_grou.repId.tolist()) # each id in one row
        with open(f'data/large/alphafold_db/train_test_val/train_group{i}.txt', 'w', encoding="utf-8") as f:
            f.write(repids)
    dfval = df[df['train1_valid2_test3'] == 2]
    dftest = df[df['train1_valid2_test3'] == 3]
    repids_val = '\n'.join(dfval.repId.tolist())
    repids_test = '\n'.join(dftest.repId.tolist())
    with open(f'data/large/alphafold_db/train_test_val/valid_all.txt', 'w', encoding="utf-8") as f:
        f.write(repids_val)
    with open(f'data/large/alphafold_db/train_test_val/test_all.txt', 'w', encoding="utf-8") as f:
        f.write(repids_test)

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

    
    
def process_pdb(pdb, saving_dir):
    path = utils.Extract_and_Save_from_PDB_with_timeout(
        input_file=pdb,
        from_dill=False,
        saving_dir=saving_dir,
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
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing for PDB files.')
    parser.add_argument('--if_allnpy_exists_delete', action='store_true', help='If enabled, rewrites concatenated .npy files. If not, uses them to create tfrecords.')
    

    args = parser.parse_args()


    pdbs = read_txt_to_list(args.pdb_ids_txt, args.input_dir, add_prefix=True)
    print(f"Found {len(pdbs)} PDB files to process.")

    s = time.time()
    ### === Using ProcessPoolExecutor for parallel processing === ###
    if args.parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            npz_paths = [executor.submit(process_pdb, pdb, args.tmp_dir) for pdb in pdbs]
            results = []
            for future in tqdm(concurrent.futures.as_completed(npz_paths), total=len(npz_paths), desc="Processing PDBs"):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except:
                    pass
    else:    
        results = []
        for pdb in tqdm(pdbs, total=len(pdbs), desc="Processing PDBs"):
            try:
                print(pdb, args.tmp_dir)
                result = process_pdb(pdb, args.tmp_dir)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {pdb}: {e}")

    FEATURES, LABELS, IDs, PLDDT = [], [], [], []
    feature_fn = os.path.join(args.tmp_dir, 'all_features.npy')
    labels_fn = os.path.join(args.tmp_dir, 'all_labels.npy')
    ids_fn = os.path.join(args.tmp_dir, 'all_ids.npy')
    plddts_fn = os.path.join(args.tmp_dir, 'all_plddts.npy')
    check_if_exist = (
        os.path.exists(feature_fn) and
        os.path.exists(labels_fn) and
        os.path.exists(ids_fn) and
        os.path.exists(plddts_fn)
        )
    if args.if_allnpy_exists_delete:
        check_if_exist = False
    if check_if_exist==False:
        features_appender = NpyAppendArray(feature_fn, delete_if_exists=True)
        labels_appender = NpyAppendArray(labels_fn, delete_if_exists=True)
        ids_appender = NpyAppendArray(ids_fn, delete_if_exists=True)
        plddt_appender = NpyAppendArray(plddts_fn, delete_if_exists=True)
        for path in tqdm(results, desc="Creating Arrays"):
            data = np.load(path)
            features = data['features']
            labels_arr = data['labels']
            ids = data['IDs']
            plddt = data['plddt']
            labels_appender.append(np.ascontiguousarray(labels_arr))
            ids_appender.append(np.ascontiguousarray(ids))
            plddt_appender.append(np.ascontiguousarray(plddt))
            features_appender.append(np.ascontiguousarray(features))
    print('loading tmp_dir/*.npy files')
    #FEATURES = np.concatenate(FEATURES, axis=0)
    FEATURES = np.load(feature_fn, mmap_mode='r')
    LABELS = np.load(labels_fn,mmap_mode='r')
    IDs = np.load(ids_fn, mmap_mode='r')
    PLDDT = np.load(plddts_fn, mmap_mode='r')

    tfmanager = utils.TFRecordManager(tfrecord_path=args.tfrecord_path, feature_dim=301, plddt=True)
    tfmanager.write_samples(features=FEATURES, labels=LABELS, ID_arrs=IDs, plddt=PLDDT)

    train_ds = tfmanager.read_dataset()
    print("Finished processing in parallel:", time.time() - s)
    for batch in train_ds:
        features, labels, ID_arrs, plddt = batch
        print(features.shape, labels.shape, ID_arrs.shape, plddt.shape if plddt is not None else "No plddt")
        print(ID_arrs)
        break


if __name__ == "__main__":
    main()
