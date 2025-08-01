import pandas as pd
import numpy as np
import dill
import os
import re
import time
import random
import string
from Bio import PDB
from Bio.PDB import Selection, PDBIO
import freesasa
from Bio.PDB.ResidueDepth import ResidueDepth, get_surface, min_dist, residue_depth
from Bio.PDB.DSSP import DSSP
from scipy.stats import spearmanr
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.vectors import Vector
from scipy.spatial import distance, distance_matrix
from constants import AMINO_ACIDS, AMINO_ACID_IDX, three_to_one, standard_amino_acids, pdb_atom_types, one_hot_encoding, one_hot_encoding_scalar
from Bio.PDB.StructureBuilder import StructureBuilder

import warnings
import gc
import psutil
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")

def remove_nonsense_atoms(residue):
    res_list = list(residue)
    nonsense = []
    for j, i in enumerate(res_list):
        if i.id not in pdb_atom_types:
            nonsense.append(i)
    return nonsense
        
        


def filter_amino_acids(input_pdb, output_pdb_filtered):
    """
    Filters out non-amino acid residues (such as water molecules, ions, and ligands)
    from a PDB file, keeping only standard amino acid residues. The filtered structure
    is saved to a new PDB file.
    Parameters:
    input_pdb (str): Path to the input PDB file.
    output_pdb_filtered (str): Path to save the filtered PDB file.
    Returns:
    None
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', input_pdb)
    io = PDB.PDBIO()
    class SelectAminoAcids(PDB.Select):
        def accept_residue(self, residue):
            # Only accept residues that are standard amino acids
            return residue.get_resname() in standard_amino_acids
    # Save the filtered structure to the output PDB file
    io.set_structure(structure)
    io.save(output_pdb_filtered, select=SelectAminoAcids())


def renumber_pdb_residues(input_pdb, output_pdb_renumbered):
    """
    Renumbers the residues in a PDB file sequentially, starting from 1. The renumbered
    structure is saved to a new PDB file.
    Parameters:
    input_pdb (str): Path to the input PDB file (typically the filtered PDB file).
    output_pdb_renumbered (str): Path to save the renumbered PDB file.
    Returns:
    None
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)
    residue_counter = 1
    for model in structure:
        for chain in model:
            for residue in chain:
                # Renumber each residue sequentially
                nonsense_atoms = remove_nonsense_atoms(residue)
                for i in nonsense_atoms: residue.detach_child(i.id)
                residue.id = (' ', residue_counter, ' ')
                residue_counter += 1
                # check for clashing atoms
                atom_list = [atom.get_coord() for atom in residue.get_unpacked_list()]
                
                atom_dist_array = distance.cdist(atom_list, atom_list)
                np.fill_diagonal(atom_dist_array, 100. )
                arg = np.argwhere(atom_dist_array < 0.01)
                if len(arg) > 0:
                    arg = arg[arg[:, 0] < arg[:, 1]]
                    for i in arg:
                        triming_at = list(residue)[np.max(i)].id
                        state_trim = 'nothing'
                        if np.max(i) > 3 and len(atom_list) > 5: 
                            residue.detach_child(list(residue)[np.max(i)].id)
                            state_trim = 'removed'
                        elif np.max(i) > 3 and len(atom_list) <= 5: 
                            list(residue)[np.max(i)].set_coord(list(residue)[np.max[i]].get_coord() + 0.07)
                            state_trim = 'moved by 0.07'
                        elif np.max(i) <= 3: 
                            list(residue)[np.max(i)].set_coord(list(residue)[np.max[i]].get_coord() + 0.02)
                            state_trim = 'moved by 0.02'
                        print(triming_at, state_trim, np.max(i))
                    
                
                
                
    # Save the renumbered structure to the output PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_renumbered)
    return structure




def renumber_pdb_residues2(input_pdb, output_pdb_renumbered):
    """
    Renumbers the residues in a PDB file sequentially, starting from 1. The renumbered
    structure is saved to a new PDB file.

    Parameters:
    input_pdb (str): Path to the input PDB file.
    output_pdb_renumbered (str): Path to save the renumbered PDB file.

    Returns:
    None
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)
    new_structure = PDB.Structure.Structure("renumbered_structure")
    residue_counter = 1

    for model in structure:
        new_model = PDB.Model.Model(model.id)
        for chain in model:
            new_chain = PDB.Chain.Chain(chain.id)
            for residue in chain:
                nonsense_atoms = remove_nonsense_atoms(residue)
                for i in nonsense_atoms: residue.detach_child(i.id)
                atom_list = [atom.get_coord() for atom in residue.get_unpacked_list()]
                atom_dist_array = distance.cdist(atom_list, atom_list)
                np.fill_diagonal(atom_dist_array, 100. )
                arg = np.argwhere(atom_dist_array < 0.01)
                if len(arg) > 0:
                    arg = arg[arg[:, 0] < arg[:, 1]]
                    for i in arg:
                        triming_at = list(residue)[np.max(i)].id
                        state_trim = 'nothing'
                        if np.max(i) > 3 and len(atom_list) > 5: 
                            residue.detach_child(list(residue)[np.max(i)].id)
                            state_trim = 'removed'
                        elif np.max(i) > 3 and len(atom_list) <= 5: 
                            list(residue)[np.max(i)].set_coord(list(residue)[np.max[i]].get_coord() + 0.07)
                            state_trim = 'moved by 0.07'
                        elif np.max(i) <= 3: 
                            list(residue)[np.max(i)].set_coord(list(residue)[np.max[i]].get_coord() + 0.02)
                            state_trim = 'moved by 0.02'
                        print(triming_at, state_trim, np.max(i))
            
            ########################
                new_residue = PDB.Residue.Residue((' ', residue_counter, residue.id[2]), 
                                                  residue.resname, residue.segid)
                atom_names = set()
                for atom in residue:
                    if atom.name in atom_names:
                        continue
                    atom_names.add(atom.name)

                    # Determine the element (if possible)
                    element = atom.element if atom.element else PDB.Element.Element(atom.name[0])

                    new_atom = PDB.Atom.Atom(atom.name, atom.coord, atom.bfactor, atom.occupancy,
                                             atom.altloc, atom.fullname, atom.serial_number, element=element)
                    new_residue.add(new_atom)

                new_chain.add(new_residue)
                residue_counter += 1
            new_model.add(new_chain)
        new_structure.add(new_model)

    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb_renumbered)

    return new_structure



def clean_and_renumber_pdb(input_pdb, output_pdb_cleaned):
    """
    Cleans a PDB file by removing all non-amino acid residues (e.g., water molecules, ions)
    and then renumbers the remaining amino acid residues sequentially. The cleaned and
    renumbered structure is saved to a new PDB file.
    Parameters:
    input_pdb (str): Path to the original input PDB file.
    output_pdb_cleaned (str): Path to save the cleaned and renumbered PDB file.
    Returns:
    None
    """
    # Step 1: Filter out non-amino acid residues
    filter_amino_acids(input_pdb, output_pdb_cleaned)
    # Step 2: Renumber residues in the filtered PDB file
    try:
        structure = renumber_pdb_residues(output_pdb_cleaned, output_pdb_cleaned)
    except:
        structure = renumber_pdb_residues2(output_pdb_cleaned, output_pdb_cleaned)
    return structure


def calc_surface_info(parsed_pdb_file, tmp_save_path, model_name):
    '''
    The input is the parsed PDB file opened via PDB.PDBParser
    The output is a SASA dict, a protein surface vetrics list dict,
    each key corresponds to results of each chain.
    '''

    SASA = {}
    SURFACE = {}
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    for model in parsed_pdb_file:
        for chain in model:
            tmp_chain = model[chain.get_id()]
            # Create a temporary PDB file for this chain
            os.makedirs(tmp_save_path, exist_ok=True)
            tmp = os.path.join(tmp_save_path, f'{model_name}_{chain.get_id()}_{random_string}.pdb')
            io = PDB.PDBIO()
            io.set_structure(chain)
            io.save(tmp)
            # caclulate SASA for temporary chain pdb
            parser = PDB.PDBParser()
            structure = parser.get_structure("protein", tmp)
            result, _ = freesasa.calcBioPDB(structure)
            SASA[chain.get_id()] = result
            # calculate residue depth and protein surface
            surface = get_surface(structure[0])
            SURFACE[chain.get_id()] = surface
            os.remove(tmp)
    return SASA, SURFACE


def normalize_feature(X, l=0, u=1):
    """
    Normalize the feature values in X using the provided formula.

    Parameters:
    X (numpy.ndarray): The feature matrix of shape (n, d), where n is the number of residues and d is the number of dimensions.
    l (float): The lower bound for normalization.
    u (float): The upper bound for normalization.

    Returns:
    numpy.ndarray: The normalized feature matrix.
    """
    X = np.array(X)
    b = np.min(X, axis=0)
    t = np.max(X, axis=0)

    X_normalized = (X - b) / ((t - b) * (u - l) + 1e-6) + l

    return X_normalized


def get_res_letter(residue):
    """
    Get the letter code for a biopython residue object
    """
    return three_to_one[residue.resname] if three_to_one[residue.resname] in AMINO_ACID_IDX else '-'


def get_side_chain_vector(residue):
    """
    Find the average of the unit vectors to different atoms in the side chain
    from the c-alpha atom. For glycine the average of the N-Ca and C-Ca is
    used.
    Returns (C-alpha coordinate vector, side chain unit vector) for residue r
    """
    gly = 0  # only becomes 1 when we have a glycine
    if not is_aa(residue) or 'CA' not in residue:  # checks weather CA is present
        return None
    ca = residue['CA'].get_coord()  # we need coords for CA as half side
    # the other half is calculated based on the average of side chain coords (pseudo position)
    side_chain_atoms = np.array([atom.get_coord() for atom in residue.get_unpacked_list()[4:]])
    # here checks weather we have a glycin or we have nothing!
    # if glycine, we get the coords of N and C instead of side chains
    if len(side_chain_atoms) < 1 and 'N' in residue and 'C' in residue:
        side_chain_atoms = np.array([residue['C'].get_coord(), residue['N'].get_coord()])
        gly = 1
    if len(side_chain_atoms) < 1:
        return None
    side_chain_atoms = side_chain_atoms - ca  # vectors pointed from ca to each side chain atom
    if gly == 1: side_chain_atoms = -side_chain_atoms  # if glycine, rotate N and C by -120
    # calculate unit vectors in vectorized manner
    magnitudes = np.sum(side_chain_atoms ** 2, axis=-1) ** (1. / 2)
    unit_vectors = side_chain_atoms / magnitudes[:, np.newaxis]
    avg_unit_vectors = unit_vectors.mean(axis=0)

    return np.array(ca), np.array(avg_unit_vectors)


def get_similarity_matrix(coords, sg=2.0, thr=1e-3):
    """
    Instantiates the distance based similarity matrix (S). S is a tuple of
    lists (I,V). |I|=|V|=|R|. Each I[r] refers to the indices
    of residues in R which are "close" to the residue indexed by r in R, and V[r]
    contains a list of the similarity scores for the corresponding residues.
    The distance between two residues is defined to be the minimum distance of
    any of their atoms. The similarity score is evaluated as
        s = exp(-d^2/(2*sg^2))
    This ensures that the range of similarity values is 0-1. sg (sigma)
    determines the extent of the neighborhood.
    Two residues are defined to be close to one another if their similarity
    score is greater than a threshold (thr).
    Residues (or ligands) for which DSSP features are not available are not
    included in the distance calculations.
    """
    sg = 2 * (sg ** 2)
    I = [[] for k in range(len(coords))]  # keeps indeces
    V = [[] for k in range(len(coords))]  # keeps similarity scores
    for i in range(len(coords)):  # for all residues search against all residues (each residue has multiple coords)
        for j in range(i, len(coords)):
            coords_i, coords_j = np.mean(coords[i], axis=0), np.mean(coords[j], axis=0)
            d = np.linalg.norm(coords_i - coords_j) # distance of COMs is taken
            #d = distance.cdist(coords[i], coords[j]).min()
            s = np.exp(-(d ** 2) / sg)  # normaliztion
            if s > thr:  # distance threshold, around 7.4< angstrom considered as neighbourhood
                I[i].append(j)
                V[i].append(s)
                if i != j:  # to avoid loop to run for all, when finds minimum between i and j, writes for both
                    I[j].append(i)
                    V[j].append(s)
    similarity_matrix = (I, V)
    
    del coords, sg, thr, i, j, d, s, I, V
    gc.collect()
    #coordinate_numbers = np.array([len(a) for a in similarity_matrix[0]])
    return similarity_matrix #, coordinate_numbers


def get_hsaac(residues, similarity_matrix):
    """
    Compute the Half sphere exposure statistics
    The up direction is defined as the direction of the side chain and is
    calculated by taking average of the unit vectors to different side chain
    atoms from the C-alpha atom
    Anything within the up half sphere is counted as up and the rest as
    down
    """
    N = len(residues)  # getting the len fo protein
    Na = len(AMINO_ACIDS)  # number of AAs = 21
    UN = np.zeros(N)  # counts the total number of neighbour AAs on upper hemisphere of each residue
    DN = np.zeros(N)  # counts the total number of neighbour AAs on downer hemisphere of each residue
    UC = np.zeros((Na, N))  # counts the type and number of
    DC = np.zeros((Na, N))

    for i, residue in enumerate(residues):
        u = get_side_chain_vector(residue)  # contains a tuple of two vectors. the first one is the CA
        if u is None:
            continue

        ca, side_chain_vector = u
        idx = AMINO_ACID_IDX[get_res_letter(residue)]  # get the index of this AA
        UC[idx, i] += 1  # count the amino acid itself as a neighbour one value for both hemispheres
        DC[idx, i] += 1  # we do that to have sum of probablitites always equall to 1, even if there is no neighbours
        neighbors = similarity_matrix[0][i]  # gets the list of indecens of neighbours in similarity matrix

        for j in neighbors:  # check  weather neighbourhood AAs are upper or downer hemisphere
            neighbor_residue = residues[j]
            if not is_aa(neighbor_residue) or not neighbor_residue.has_id('CA'):
                continue

            neighbor_ca = np.array(neighbor_residue['CA'].get_coord())  # get CA pf neighbour AA
            angle = np.arccos(angle_between_vectors(side_chain_vector, neighbor_ca - ca))  # calculate the angle between neighbour CA and CA of current residue
            neighbor_idx = AMINO_ACID_IDX[get_res_letter(neighbor_residue)]  # get the index of neighbour AA

            if angle < np.pi / 2.:
                UN[i] += 1
                UC[neighbor_idx, i] += 1
            else:
                DN[i] += 1
                DC[neighbor_idx, i] += 1

    UC /= (1.0 + UN)
    DC /= (1.0 + DN)
        
    # Clean up all unnecessary variables to free memory
    del residues, similarity_matrix, N, Na, u, ca, side_chain_vector, idx, neighbors
    del i, j, neighbor_residue, neighbor_ca, angle, neighbor_idx
    
    # Force garbage collection to clean up cyclic references
    gc.collect()

    return UC, DC, UN, DN



def get_hsaac_for_pdb_residues(structure, total_residues):
    #total_residues = sum(len(list(chain.get_residues())) for model in structure for chain in model)
    output = np.zeros((total_residues, 2 * len(AMINO_ACIDS)))
    out_un = np.zeros(total_residues)
    out_dn = np.zeros(total_residues)
    
    index = 0
    for model in structure:
        for chain in model:
            residues = list(chain.get_residues())
            n_residues = len(residues)
            coords = [[atom.get_coord() for atom in residue.get_atoms()] for residue in residues]
            similarity_matrix = get_similarity_matrix(coords)
            UC, DC, UN, DN = get_hsaac(residues, similarity_matrix)
            hsaacs = np.concatenate((UC, DC), axis=0).T # Transpose to get N x (2 * Na) matrix
            
            output[index:index+n_residues] = hsaacs
            out_un[index:index+n_residues] = UN
            out_dn[index:index+n_residues] = DN
            
            index += n_residues
            
            del residues, coords, similarity_matrix, UC, DC, UN, DN, hsaacs
            gc.collect()
    
    return output, out_un, out_dn


'''
def get_hsaac_for_pdb_residues(structure):
    output = []
    out_un = []
    out_dn = []
    for model in structure:
        for chain in model:
            residues = [residue for residue in chain.get_residues()]
            coords = [[atom.get_coord() for atom in residue.get_atoms()] for residue in residues]
            similarity_matrix = get_similarity_matrix(coords)
            UC, DC, UN, DN = get_hsaac(residues, similarity_matrix)
            hsaacs = np.concatenate((UC, DC), axis=0).T  # Transpose to get N x (2 * Na) matrix
            output.append(hsaacs)
            out_un.append(UN)
            out_dn.append(DN)
            del residues, coords, similarity_matrix, UC, DC, UN, DN, hsaacs
            gc.collect()
    return np.concatenate(output), np.concatenate(out_un), np.concatenate(out_dn)
'''

def Protrusion_index(array, res_labels=0, V_atom=20.1, radius=10):
    '''
    receives an array with shape (N, 3) and calculates the porusion index for each atom in it.
    portusion index is the ratio between free and occupied space in certain radius of each atom.
    formula: external_vol/internal_vol. where external is free and interal is occupied space.
    V_atom is considered as the constant volume of occupation of each atom around probe atom.

    If res_labels is provided, the average, max, mean and stddev of protrusions per residues

    returns a dict that contains the asked arrays in shape (N,)
    '''

    output = {}
    distances = distance.cdist(array, array)  # 2D array with shape (N,N) maps all distances between all atoms
    N_atoms = np.where(distances <= radius, 1, 0)  # 2D array that maps which distances are below radius and which are not
    N_atoms = np.sum(N_atoms, axis=0)  # 1D array with shape (N,) that counts all atoms with distance less than radius
    V_spheres = (4 / 3) * np.pi * (radius ** 3)
    maximum_Cx = (V_spheres - V_atom) / V_atom # a single value, for normalization
    V_spheres = np.array([V_spheres] * array.shape[0])  # vectorizes all the spheres (same for all atoms) 1D
    V_atoms = np.array([V_atom] * array.shape[0])  # vectorizes all the V_atoms (same for all atoms) 1D

    # calculate internal volumes = N_atoms * V_atoms
    V_ints = np.multiply(N_atoms, V_atoms)  # 1D array (N,)
    # calculate external volumes = V_spheres - V_ints
    V_exts = V_spheres - V_ints  # 1D array (N,)
    # calculate protrusion indexes: V_exts/V_ints
    Cx = V_exts / V_ints  # 1D array (N,)
    output.update({'cx': Cx / maximum_Cx})
    if type(res_labels) == type(Cx):
        if res_labels.shape[0] != array.shape[0]: raise ValueError(
            'res label and given array do not have the same shape')
        # need to first get inverese indeces which is simply a 1D array of e.g [0,0,0,1,1,1,1,2,2,2,3,3,3,3], just starts labels from 0
        unique_labels, inverse_indeces = np.unique(res_labels, return_inverse=True)
        # np.bincount computes the sums of values with similar labels,weights are Cx values
        sums = np.bincount(inverse_indeces, weights=Cx)  # dimension is (number_of_labels, )
        counts = np.bincount(inverse_indeces)  # dimension is (number_of_labels, )
        ### AVERAGE CALCULATION
        means = sums / counts  # dimension is (number_of_labels, )
        normin = means / maximum_Cx # normalized by maximum value
        cx_avg = normin[inverse_indeces]  # assigns the mean values in cx_avg to their corresponding positions based on stored labels in inverse_indeces
        ### MAXIMUM CALCULATION
        cx_max = np.zeros_like(unique_labels,
                               dtype=Cx.dtype)  # this is 1D (number_of_labels, ) zero for storing maximums in
        # np.maximum.at computes maximum values for each bin. it updates the cx_max array at the positions specified by inverse_indices with the corresponding values from Cx
        np.maximum.at(cx_max, inverse_indeces, Cx)  # output, positions, values (1D array (number_of_labels, )), (N, ), (N, )
        cx_max = cx_max / maximum_Cx
        cx_max = cx_max[inverse_indeces]  # mapping back to original bin positions (N, )
        ### MINIMUM CALCULATION
        cx_min = np.zeros_like(unique_labels, dtype=Cx.dtype) + 100000.  # 1D (number_of_labels, ) with 100000 values in each index
        np.minimum.at(cx_min, inverse_indeces, Cx)
        cx_min = cx_min / maximum_Cx
        cx_min = cx_min[inverse_indeces]
        ### STDEV CALCULATION
        squared_diffs = (Cx - means[inverse_indeces]) ** 2  # squared difference from mean (N,)
        squared_diffs_sum = np.bincount(inverse_indeces,
                                        weights=squared_diffs)  # sum of all squared diffs for each bin (number_of_labels, )
        std_devs = np.sqrt(squared_diffs_sum / counts)
        std_devs = std_devs / (maximum_Cx/2) # maximum stddev is equal to mean of minimum and maximum values
        cx_std = std_devs[inverse_indeces]
        output.update({'cx_avg': cx_avg, 'cx_max': cx_max, 'cx_min': cx_min, 'cx_std': cx_std})
    del distances, N_atoms, V_spheres, maximum_Cx, V_atoms, V_ints, V_exts, Cx, cx_avg, cx_max, cx_min, std_devs, squared_diffs
    gc.collect()
    return output


def angle_between_vectors(a, b):

    dot_prod = np.dot(a, b)
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)
    cos = dot_prod / (a_mag * b_mag)
    return cos





def vectorized_cos_sin_between_vectors(a, b):
    # Normalize the vectors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a_mag = a / np.linalg.norm(a, axis=1)[:, np.newaxis]
        b_mag = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    # Calculate the cosine of the angle
    cos = np.einsum('ij,ij->i', a_mag, b_mag)
    # Replace NaN values with 0 in the cosine
    cos = np.nan_to_num(cos, nan=0.0)
    # Calculate the cross product
    cross_product = np.cross(a_mag, b_mag)
    # Calculate the sine of the angle using the cross product
    sin = np.linalg.norm(cross_product, axis=1)
    # Determine the sign of the sine using the cross product's direction
    sin_sign = np.sign(cross_product[:, 2])  # Use z-component for sign
    sin *= sin_sign
    # Replace NaN values with 0 in the sine
    sin = np.nan_to_num(sin, nan=0.0)
    del a_mag, b_mag
    gc.collect()

    return cos, sin

def vectorized_dihedral_sin_cosine(ca1_to_beta1, beta1_to_beta2, beta2_to_back2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        b1 = ca1_to_beta1#(N, 3)
        b2 = beta1_to_beta2#(N, 3)
        b3 = beta2_to_back2#(N, 3)
        # Cross products to get prependicular vectors
        n1 = np.cross(b1, b2)#(N, 3)
        n2 = np.cross(b2, b3)#(N, 3)
        # normalize prependicular vectors
        n1 /= np.linalg.norm(n1, axis=1)[:, np.newaxis] #(N, 3)
        n2 /= np.linalg.norm(n2, axis=1)[:, np.newaxis] #(N, 3)
        # since b2 is our central axis and needed for sin calc, we normalize it too
        b2_unit = b2 / np.linalg.norm(b2, axis=1)[:, np.newaxis]
        # Cosine and sine of the dihedral angles
        cos = np.einsum('ij,ij->i', n1, n2)
        sin = np.einsum('ij,ij->i', np.cross(n1,n2), b2_unit)
        cos = np.nan_to_num(cos, nan=0.0)
        sin = np.nan_to_num(sin, nan=0.0)
    del b1, b2, b3, n1, n2, b2_unit
    gc.collect()
    return cos, sin
    
    


def residue_geometric_centroid(residue):
    side_chain_atoms = np.array([atom.get_coord() for atom in residue.get_unpacked_list()[4:]])
    if len(side_chain_atoms) < 1 and 'CA' in residue and 'O' in residue: # for glycine
        side_chain_atoms = np.array([residue['CA'].get_coord(), residue['O'].get_coord()])
    if len(side_chain_atoms) < 1 and 'CA' in residue and 'C' in residue:
        side_chain_atoms = np.array([residue['CA'].get_coord(), residue['C'].get_coord()])
    if len(side_chain_atoms) < 1 and 'N' in residue and 'CA' in residue:
        side_chain_atoms = np.array([residue['CA'].get_coord(), residue['N'].get_coord()])
    if len(side_chain_atoms) < 1 and 'CA' in residue:
        side_chain_atoms = np.array([residue['CA'].get_coord(), residue['CA'].get_coord()]) + 0.1
    centroid = np.mean(side_chain_atoms, axis=0)
    return centroid

def residue_backbone_centroid(residue):
    side_chain_atoms = np.array([atom.get_coord() for atom in residue.get_unpacked_list()[:3]])
    centroid = np.mean(side_chain_atoms, axis=0)
    return centroid

def get_unique_df(df, subset='residue', keep='first'):
    df_unique = df.drop_duplicates(subset=subset, keep=keep) # Drop duplicates based on the 'Label' column, keeping the first occurrence
    df_unique = df_unique.reset_index(drop=True) # Reset index if needed
    return df_unique


def self_distance_matrix(array, k_nearest=1):
    """
    Computes the k-nearest neighbors for each point in the input array based on the Euclidean distance matrix.
    Excludes each point itself and its immediate ±2 neighbors when determining the nearest neighbors.

    Parameters:
    array (ndarray): A 2D array where each row represents a point in a multi-dimensional space.
    k_nearest (int): Number of nearest neighbors to return.

    Returns:
    tuple: Two numpy arrays of shape (n_points, k_nearest):
        - nearest_neighbours (ndarray): Indices of the k-nearest neighbors for each point.
        - nearest_distances (ndarray): Distances to the k-nearest neighbors for each point.
    """
    dists = distance.cdist(array, array)# Calculate the pairwise distance matrix
    np.fill_diagonal(dists, 1000.)# Set the diagonal to a large value (e.g., 1000) to ignore self-distances
    s = dists.shape[0] # Get the number of points (rows/columns in the distance matrix)
    # Create masks to ignore diagonal, previous, and next indices
    mask0 = np.eye(s, k=0, dtype=bool)  # Diagonal (self-distance)
    mask1 = np.eye(s, k=1, dtype=bool)  # Next index
    mask2 = np.eye(s, k=-1, dtype=bool)  # Previous index
    mask3 = np.eye(s, k=2, dtype=bool)
    mask4 = np.eye(s, k=-2, dtype=bool)
    mask = mask0 | mask1 | mask2 | mask3 | mask4
    #mask = np.where(mask == 1, True, False)# Convert the mask to a boolean array where True indicates positions to ignore
    dists[mask] = 1000.# Apply the mask to the distance matrix, setting masked positions to a large value
    
    # Get k smallest distances and corresponding indices
    nearest_indices = np.argpartition(dists, kth=k_nearest, axis=1)[:, :k_nearest]
    nearest_dists = np.take_along_axis(dists, nearest_indices, axis=1)
    # Sort by actual distances
    sort_order = np.argsort(nearest_dists, axis=1)
    nearest_indices = np.take_along_axis(nearest_indices, sort_order, axis=1)
    nearest_dists = np.take_along_axis(nearest_dists, sort_order, axis=1)
    del mask0, mask1, mask2, mask3, mask4, mask, dists, s
    gc.collect()
    return nearest_indices, nearest_dists


def array_to_array_vectors(arr1, arr2):
    """
    Computes the difference vectors between all points in arr1 and all points in arr2.
    Parameters:
    arr1 (ndarray): The first 2D array where each row represents a point in a multi-dimensional space.
    arr2 (ndarray): The second 2D array with the same dimensions as arr1.
    Returns:
    ndarray: A 3D array where the element [i, j, :] represents the difference vector between
             point i in arr1 and point j in arr2 (i.e., arr1[i] - arr2[j]).
    """
    # Ensure both arrays have the same shape
    if arr1.shape[1] != arr2.shape[1]:
        raise ValueError("Both input arrays must have the same number of dimensions (columns).")
    # Expand dimensions to enable broadcasting
    arr1_expanded = arr1[:, np.newaxis, :]  # Shape: (N1, 1, D)
    arr2_expanded = arr2[np.newaxis, :, :]  # Shape: (1, N2, D)
    # Compute the difference vectors using broadcasting
    diff_vectors = arr1_expanded - arr2_expanded  # Shape: (N1, N2, D)
    del arr1_expanded, arr2_expanded
    gc.collect()
    return diff_vectors


def retrieve_plus_minus_vectors(diff_vectors):
    """
    Retrieves n-1 and n+1 difference vectors for each coordinate.
    Parameters:
    diff_vectors (ndarray): A 3D array of difference vectors (S, S, D).
    Returns:
    ndarray: A 3D array (S, 2, D) containing n-1 and n+1 difference vectors.

    """
    S, _, D = diff_vectors.shape
    result = np.zeros((S, 2, D)) # (number of aas, n-1 and n+1, xyz coords)
    # retreive n-1 differences for each index
    mask0 = np.eye(S, k=-1, dtype=bool)
    result[1:, 0, :] = diff_vectors[mask0]
    # retreive n+1 differences for each index
    mask1 = np.eye(S, k=1, dtype=bool)
    result[:-1, 1, :] = diff_vectors[mask1]
    del S, D, mask0, mask1
    gc.collect()
    return -result

def retrieve_plus_minus_vectors_of_nearest_neighbour(diff_vectors, nearest_neighbour_indeces):
    '''
    Args:
        diff_vectors: np.ndarray of shape (N, N, D)
        nearest_neighbour_indeces: np.ndarray of shape (N,)
    Returns:
        result_minus, result_self, result_after: Each (N, 3, D) tensors
        with masks applied where applicable
    '''
    N, _, D = diff_vectors.shape
    # extract diff vectors for each i, the minus, self (which is i itself) and plus
    indexes_self = np.arange(N) #[0, ..., n-1]
    i_plus_indexes = indexes_self + 1
    i_plus_indexes[-1] = i_plus_indexes[-1] - 1 # just because last index+1 is out of range, therefore use the last index instead. it will be masked later because it does not exist.

    j_plus_indexes = nearest_neighbour_indeces + 1
    j_plus_indexes = np.where(j_plus_indexes >= len(j_plus_indexes), nearest_neighbour_indeces, j_plus_indexes) # if j_plus_indexes is larger than indeces, use the nearest neighbour index itself.

    j_minus_indexes = nearest_neighbour_indeces - 1
    j_minus_indexes = np.where(j_minus_indexes < 0, nearest_neighbour_indeces, j_minus_indexes) # if j-1 does not exist, therefore, use j itself. will be masked.

    ## diff vecs
    i_minus_diffvec = diff_vectors[indexes_self-1, :, :] #[1:] First one has no previous index, so should be excluded #(N,N,3) for each n, n+1 diff vector is at its index dim 0
    i_minus_diffvec[0] = diff_vectors[0]

    i_plus_diffvec = diff_vectors[i_plus_indexes, :, :] #[:-1] Last one has no plus index, so should be excluded #(N,N,3) for each n, n+1 diff vector is at its index dim 0. for last one, n is taken and will be masked.

    assert indexes_self.shape == j_minus_indexes.shape == (N,)
    # Now, for each i_minus, i and i_plus, we should get the nearest neighbour to i, which is called j, and take its i-1 -> [j-1, j, j+1], i -> [j-1, j, j+1] and i+1 -> [j-1, j, j+1]
    # we should make sure that if j-1 or j+1 do not exist (first or last amino acid), they should be masked
    ### i_minus versus all js
    i_minus_j_minus_diffvec = i_minus_diffvec[indexes_self, j_minus_indexes, :][:, np.newaxis, :] # (N, 1, 3)
    i_minus_j_minus_mask = np.where(nearest_neighbour_indeces-1 < 0, 1., 0.) # if no previous index is there (for j), it should get masked with token == 1
    i_minus_j_minus_mask[0] = 1. # zero index should be masked, because i-1 for i==0 does not exist!

    i_minus_j_self_diffvec = i_minus_diffvec[indexes_self, nearest_neighbour_indeces, :][:, np.newaxis, :]  # (N, 1, 3)
    i_minus_j_self_mask = np.ones_like(i_minus_j_minus_mask) # it is basically no mask, because nearest neighbour always exists.
    i_minus_j_self_mask[0] = 1. # Again, zero index should be masked, because i-1 for i==0 does not exist!

    i_minus_j_plus_diffvec = i_minus_diffvec[indexes_self, j_plus_indexes, :][:, np.newaxis, :]
    i_minus_j_plus_mask = np.where(nearest_neighbour_indeces+1 >= len(nearest_neighbour_indeces), 1., 0.) # if no plus index exists for j, mask it
    i_minus_j_plus_mask[0] = 1. # zero index should be masked, because i-1 for i==0 does not exist!

    ### i_self versus all js
    i_self_j_minus_diffvec = diff_vectors[indexes_self, j_minus_indexes, :][:, np.newaxis, :] #(N,1,3)
    i_self_j_minus_mask = np.where(nearest_neighbour_indeces-1 < 0, 1., 0.) # index 0 does not get masked because we have self i, not i-1

    i_self_j_self_diffvec = diff_vectors[indexes_self, nearest_neighbour_indeces, :][:, np.newaxis, :]
    i_self_j_self_mask = np.ones_like(i_self_j_minus_mask)

    i_self_j_plus_diffvec = diff_vectors[indexes_self, j_plus_indexes, :][:, np.newaxis, :]
    i_self_j_plus_mask = np.where(nearest_neighbour_indeces+1 >= len(nearest_neighbour_indeces), 1., 0.) # if no plus index exists for j, mask it

    
    ### i_plus versus all js:
    i_plus_j_minus_diffvec = i_plus_diffvec[indexes_self, j_minus_indexes, :][:, np.newaxis, :]
    i_plus_j_minus_mask = np.where(nearest_neighbour_indeces-1 < 0, 1., 0.) # if j-1 does not exists mask it
    i_plus_j_minus_mask[-1] = 1. # last index is maked because no plus index exists for it (for i, not j)

    i_plus_j_self_diffvec = i_plus_diffvec[indexes_self, nearest_neighbour_indeces, :][:, np.newaxis, :]
    i_plus_j_self_mask = np.ones_like(i_plus_j_minus_mask) # all js exist
    i_plus_j_self_mask[-1] = 1. # last i, does not have i+1

    i_plus_j_plus_diffvec = i_plus_diffvec[indexes_self, j_plus_indexes, :][:, np.newaxis, :]
    i_plus_j_plus_mask = np.where(nearest_neighbour_indeces+1 >= len(nearest_neighbour_indeces), 1., 0.) # where j+1 does not exist
    i_plus_j_plus_mask[-1] = 1. # last i+1 does not exist

    assert i_plus_j_minus_diffvec.shape == i_self_j_minus_diffvec.shape == i_minus_j_minus_diffvec.shape == (N,1,D), f"Shapes do not match: {i_plus_j_minus_diffvec.shape}, {i_self_j_minus_diffvec.shape}, {i_minus_j_minus_diffvec.shape}"
    assert i_plus_j_self_diffvec.shape == i_self_j_self_diffvec.shape == i_minus_j_self_diffvec.shape == (N,1,D), f"Shapes do not match: {i_plus_j_self_diffvec.shape}, {i_self_j_self_diffvec.shape}, {i_minus_j_self_diffvec.shape}"
    assert i_plus_j_plus_diffvec.shape == i_self_j_plus_diffvec.shape == i_minus_j_plus_diffvec.shape == (N,1,D), f"Shapes do not match: {i_plus_j_plus_diffvec.shape}, {i_self_j_plus_diffvec.shape}, {i_minus_j_plus_diffvec.shape}"
    # assert mask shapes 
    assert i_minus_j_minus_mask.shape == i_minus_j_self_mask.shape == i_minus_j_plus_mask.shape == (N,), f"Shapes do not match: {i_minus_j_minus_mask.shape}, {i_minus_j_self_mask.shape}, {i_minus_j_plus_mask.shape}"
    assert i_self_j_minus_mask.shape == i_self_j_self_mask.shape == i_self_j_plus_mask.shape == (N,), f"Shapes do not match: {i_self_j_minus_mask.shape}, {i_self_j_self_mask.shape}, {i_self_j_plus_mask.shape}"
    assert i_plus_j_minus_mask.shape == i_plus_j_self_mask.shape == i_plus_j_plus_mask.shape == (N,), f"Shapes do not match: {i_plus_j_minus_mask.shape}, {i_plus_j_self_mask.shape}, {i_plus_j_plus_mask.shape}"

    result_minus = np.concatenate([i_minus_j_minus_diffvec, i_minus_j_self_diffvec, i_minus_j_plus_diffvec], axis=1) # (N,3,3) [i-1-->j-1, i-1-->j, i-1-->j+1]
    result_self = np.concatenate([i_self_j_minus_diffvec, i_self_j_self_diffvec, i_self_j_plus_diffvec], axis=1) # (N,3,3) [i-->j-1, i-->j, i-->j+1]
    result_after = np.concatenate([i_plus_j_minus_diffvec, i_plus_j_self_diffvec, i_plus_j_plus_diffvec], axis=1) #(N,3,3) [i+1--j-1, i+1-->j, i+1-->j+1]

    return result_minus, result_self, result_after


def find_inter_residues(arr1, lengths, inter_threshold=7):
    """
    Identifies inter-group residue pairs within a 2D array based on a distance threshold,
    excluding intra-group comparisons.
    Parameters:
    arr1 (numpy.ndarray): A 2D array of shape (S, D) containing the coordinates or features of residues.
    lengths (list of int): A list specifying the sizes of groups within arr1, used to determine intra-group boundaries.
    inter_threshold (float, optional): The distance threshold for considering pairs as 'close'. Default is 7.
    Returns:
    inter_pairs (numpy.ndarray): An array of shape (N, 2) containing indices of pairs where the distance is
                                 below or equal to the threshold, excluding intra-group pairs.
    inter_nonpairs (numpy.ndarray): An array of shape (M, 2) containing indices of pairs where the distance
                                    is above the threshold, excluding intra-group pairs.
    """
    # Calculate the pairwise distance matrix between all elements in arr1
    dists = distance.cdist(arr1, arr1)
    # Create boolean arrays where True indicates distances below or above the threshold
    dists_pair = np.where(dists <= inter_threshold, True, False)
    dists_nonpair = np.where(dists > inter_threshold, True, False)
    # Initialize a mask to identify intra-group pairs (diagonal and within groups)
    s = arr1.shape[0]
    mask = np.eye(s, k=0, dtype=bool)
    i = 0
    for l in lengths:
        # Mark intra-group indices in the mask as True
        mask[i:l + i, i:l + i] = True
        i = i + l
    # Apply mask to exclude intra-group pairs from consideration
    dists_pair[mask] = False
    dists_nonpair[mask] = False
    # Find indices of pairs where the distance is below the threshold
    inter_pairs = np.argwhere(dists_pair == True)
    # Filter to keep only one of each symmetric pair (i < j)
    inter_pairs = inter_pairs[inter_pairs[:, 0] < inter_pairs[:, 1]]
    # Find indices of pairs where the distance is above the threshold
    inter_nonpairs = np.argwhere(dists_nonpair == True)
    # Filter to keep only one of each symmetric pair (i < j)
    inter_nonpairs = inter_nonpairs[inter_nonpairs[:, 0] < inter_nonpairs[:, 1]]
    # Return the pairs with distances below and above the threshold
    del dists, dists_pair, dists_nonpair, mask
    gc.collect()
    return inter_pairs, inter_nonpairs

def read_dill_and_get_dfs(dill_file):
    with open(dill_file, 'rb') as f:
        data = dill.load(f)
    L = []
    for i in data:
        if type(i) == type(pd.DataFrame({'amir': [1, 2, 3]})): L.append(i)
    L = pd.concat(L).reset_index(drop=True)
    return L



def get_df_from_dill(input):
    with open(input, 'rb') as f:
        data = dill.load(f)
    df = data['df']
    del data
    gc.collect()
    return df

def load_data_from_df(df, cols=None):
    J = False
    if cols is None:
        J = True
        cols = ['t25_cos', 't85_cos', 'dihedral_cos', 'dihedral_sin', 'nearest_distances',
                'UN', 'DN', 'cx_avg', 'res_mapped', 'nearest_neighbours']
    df_unique = df.drop_duplicates(subset='residue', keep='first')  # keep only one row per residue
    df_unique = df_unique.reset_index(drop=True)
    df_unique['res_mapped'] = df_unique['resname'].map(one_hot_encoding_scalar).fillna(-1)  # one hot encode
    df_unique['res_mapped'] = df_unique['res_mapped'].astype(int)
    df_unique['nearest_neighbours'] = df_unique['nearest_neighbours'].astype(int)
    # Ensure 'cols' are present in the DataFrame
    if not set(cols).issubset(df_unique.columns):
        raise ValueError("Some columns in 'cols' are not present in the DataFrame")
    array = df_unique[cols].to_numpy()
    if J:
        labs = array[:, -1].astype(int)
        array[:, -1] = array[labs, -2]
        # remove those with unknown residue that are filled with -1
        mask = (array[:, -1] != -1) & (array[:, -2] != -1)
        #print(mask)
        array = array[mask]
    return array



############# General Functions

def df_to_pdb(df, default_cols=True, cols=None, filename='tmp.pdb'):
    if default_cols:
        cols = {'chainID':'chain', 'resName':'resname',
                'x':'x', 'y':'y', 'z':'z', 'resid':'residue',
                'element':'element', 'atomname':'atom_name'}
    else: cols = cols
    #chains = df.drop_duplicates(subset=cols['chainID'], keep='first')[cols['chainID']].tolist()

    ATOM, altLoc, iCode, occupancy, tempFactor, charge = 'ATOM', ' ', ' ', 1, 0.00, ' '
    with open(filename, 'w') as f:
        previous_chain = None
        previous_res = None
        res_number = 0
        prev_resname = None
        for index, row in df.iterrows():
            if row[cols['atomname']] not in pdb_atom_types: continue
            # Check if the chain has changed
            if previous_chain is not None and previous_chain != row[cols['chainID']]:
                # Write the TER line for the previous chain
                f.write(f"TER   {(index):>5}      {prev_resname} {previous_chain:<1}{res_number:>4}\n")
            if row[cols['resid']] != previous_res:
                res_number = res_number + 1

            f.write(f"{ATOM:<6}{(index+1):>5} "
                    f"{row[cols['atomname']]:<4}{altLoc:<1}"
                    f"{row[cols['resName']]:<3} {row[cols['chainID']]:<1}"
                    f"{res_number:>4}{iCode:<1}   "
                    f"{row[cols['x']]:>8.3f}{row[cols['y']]:>8.3f}{row[cols['z']]:>8.3f}"
                    f"{occupancy:>6.2f}{tempFactor:>6.2f}          "
                    f"{row[cols['element']]:>2}{charge:>2}\n")

            previous_res = row[cols['resid']]  # change previous res id
            previous_chain = row[cols['chainID']] # Update the previous_chain to the current chain
            prev_resname = row[cols['resName']]
        f.write(f"TER   {(index+1):>5}      {row[cols['resName']]} {row[cols['chainID']]:<1}{res_number:>4}\n")
        f.write("END\n")
        f.close()




def pdb_to_df(pdb_file: str, structure, add_sasa=True, add_cx=True, pdb_id='A'):
    # Parse PDB
    os.makedirs('tmp', exist_ok=True)
    if add_sasa: sasa, surface = calc_surface_info(structure, 'tmp', pdb_id)
    atom_data = []  # keeps dictionaries, each a row of df
    model_chains = []
    for model in structure:
        res_id = 0
        for chain in model:
            model_chains.append(chain.get_id())
            a = 0
            for residue in chain:
                for atom in residue:
                    sidechain_pseudopos = residue_geometric_centroid(residue)

                    backbone_pseudopos = residue_backbone_centroid(residue)
                    atom_info = {
                                 'model': model.get_id(),
                                 'chain': chain.get_id(),
                                 'residue': int(residue.get_id()[1]),
                                 'resname': residue.get_resname(),
                                 'x': atom.coord[0],
                                 'y': atom.coord[1],
                                 'z': atom.coord[2],
                                 'element': atom.element,
                                 'atom_name': atom.get_name(),
                                 'aid': atom.get_serial_number(),
                                 'res_id': res_id,
                                 'sidechain_pseudopos_x':sidechain_pseudopos[0],
                                 'sidechain_pseudopos_y': sidechain_pseudopos[1],
                                 'sidechain_pseudopos_z': sidechain_pseudopos[2],
                                 'backbone_pseudopos_x':backbone_pseudopos[0],
                                 'backbone_pseudopos_y':backbone_pseudopos[1],
                                 'backbone_pseudopos_z':backbone_pseudopos[2]}
                    if add_sasa:
                        atom_info.update({'sasa': sasa[chain.get_id()].atomArea(a - 1) / 75}) # approximate maximum value
                        atom_info.update({'rd_value': min_dist(atom.coord, surface[chain.get_id()]) / 7.5}) # approximate maximum value
                    a = a + 1
                    atom_data.append(atom_info)
                res_id = res_id + 1
    df = pd.DataFrame(atom_data)
    if add_cx:
        DICTS = []
        for ch in model_chains:
            df1 = df[df['chain'] == ch]
            xyz_arr = df1[['x', 'y', 'z']].to_numpy()
            res_labels = df1['res_id'].to_numpy()
            cx_dict = Protrusion_index(xyz_arr, res_labels, V_atom=20.1, radius=10)
            DICTS.append(pd.DataFrame(cx_dict))
        DICTS = pd.concat(DICTS)
        for col in DICTS.columns: df[col] = DICTS[col].tolist()
    if add_sasa: del sasa, surface
    if add_cx: del cx_dict, xyz_arr
    del structure
    gc.collect()
    return df, model_chains


def compute_geometrical_features(sidechain_centroid_array, backbone_centroid_array, res_labels, k_nearest=1):

    #### COMPUTE VECTORS
    # nearest neighbours are defined by side chains and backbone centroids separately
    nearest_neighbours_cb_all, nearest_distances_cb_all = self_distance_matrix(sidechain_centroid_array, k_nearest=k_nearest) #(N, k_nearest)
    nearest_neighbours_ca_all, nearest_distances_ca_all = self_distance_matrix(backbone_centroid_array, k_nearest=k_nearest)
    sidechains_diff_vecs = array_to_array_vectors(sidechain_centroid_array, sidechain_centroid_array)
    backbone_diff_vecs = array_to_array_vectors(backbone_centroid_array, backbone_centroid_array)
    backside_diff_vecs = array_to_array_vectors(sidechain_centroid_array, backbone_centroid_array)

    ## Define backbone vectors of the ith amino acid which are always the same and do not depend on the nearest neighbours
    #> (u1, u3) --> CAi-1---CA---CAi+1 
    # direction is = (CAi-1 - CAi) and (CAi+1 - CAi) i-->i+1 or i-1
    u1u3 = retrieve_plus_minus_vectors(backbone_diff_vecs) # (N, 2, 3) with (,0,) for ni-1 and (,1,) for ni+1
    #> u2 --> CAi--CBi , direction is CA --> CB
    mask = np.eye(backside_diff_vecs.shape[0], k=0, dtype=bool)
    u2 = backside_diff_vecs[mask] # (N, 3) carries differences for each
    #> (u4, u6)  --> CBi-1--CBi--CBi+1 and also for the nearest neighbour
    # direction is = (CBi-1 - CBi) and (CBi+1 - CBi)
    u4u6 = retrieve_plus_minus_vectors(sidechains_diff_vecs)  # (N, 2, 3) with (,0,) for ni-1 and (,1,) for ni+1
    # for each k_nearest neighbour we need to retrieve the vectors for each residue
    for k in range(k_nearest):
        nearest_neighbours_cb = nearest_neighbours_cb_all[:, k] # kth nearest neighbour (N,) indeces from cb
        nearest_distances_cb = nearest_distances_cb_all[:, k] # (N,) with distances to the nearest neighbours
        nearest_neighbours_ca = nearest_neighbours_ca_all[:, k] # from ca
        nearest_distances_ca = nearest_distances_ca_all[:, k] 
        # defining the vectors:
        #(u10, u11) same as u1u3 but for kth nearest neighbour --> CAj-1---CA---CAj+1  it is the neaest neighbour backbone and is called j.
        u10u11 = u1u3[nearest_neighbours_cb] # (N, 2, 3) with (,0,) for nj-1 and (,1,) for nj+1
        #u2 --> CAj--CBj , direction is CA --> CB
        u8 = u2[nearest_neighbours_cb] # (N, 2) carries differences for each nearest neighbour
        # (u7, u9) --> CBj-1--CBj--CBj+1 and also for the nearest neighbour
        # direction is = (CBj-1 - CBj) and (CBj+1 - CBj)
        u7u9 = u4u6[nearest_neighbours_cb]  # (N, 2, 3) with (,0,) for nj-1 and (,1,) for nj+1
        ## u5, which is CBi----CBj
        u5 = sidechains_diff_vecs[np.arange(sidechains_diff_vecs.shape[0]), nearest_neighbours_cb] # (N, 3) dim
        ## define additional vectors that were added later:
        # u12,...,u19 are vectors between CBis and CBjs

        #### COMPUTE ANGLE COSINES
        t12 = vectorized_cos_sin_between_vectors(u1u3[:, 0, :], u2)
        t23 = vectorized_cos_sin_between_vectors(u2, u1u3[:, 1, :])
        t45 = vectorized_cos_sin_between_vectors(u4u6[:, 0, :], u5)
        t56 = vectorized_cos_sin_between_vectors(u5, u4u6[:, 1, :])
        t78 = vectorized_cos_sin_between_vectors(u7u9[:, 0, :], u8)
        t89 = vectorized_cos_sin_between_vectors(u8, u7u9[:, 1, :])
        t75 = vectorized_cos_sin_between_vectors(u7u9[:, 0, :], u5)
        t95 = vectorized_cos_sin_between_vectors(u7u9[:, 1, :], u5)
        t42 = vectorized_cos_sin_between_vectors(u4u6[:, 0, :], u2)
        t62 = vectorized_cos_sin_between_vectors(u4u6[:, 1, :], u2)
        t108 = vectorized_cos_sin_between_vectors(u10u11[:, 0, :], u8)
        t811 = vectorized_cos_sin_between_vectors(u8, u10u11[:, 1, :])
        t110 = vectorized_cos_sin_between_vectors(u1u3[:, 0, :], u10u11[:, 0, :])
        t311 = vectorized_cos_sin_between_vectors(u1u3[:, 1, :], u10u11[:, 1, :])
        t28 = vectorized_cos_sin_between_vectors(u2, u8)
        t47 = vectorized_cos_sin_between_vectors(u4u6[:, 0, :], u7u9[:, 0, :])
        t69 = vectorized_cos_sin_between_vectors(u4u6[:, 1, :], u7u9[:, 1, :])
        t25 = vectorized_cos_sin_between_vectors(u2, u5)
        t85 = vectorized_cos_sin_between_vectors(u8, u5)
        dihedral_cos, dihedral_sin = vectorized_dihedral_sin_cosine(ca1_to_beta1=u2, beta1_to_beta2=-u5, beta2_to_back2=-u8)
    

        # reverese_mapping
        unique_labels, inverse_indeces = np.unique(res_labels, return_inverse=True)
        # final dict
        final_dict = {
            f'nearest_neighbours_cb_{k}': nearest_neighbours_cb[inverse_indeces],
            f'nearest_distances_cb_{k}': nearest_distances_cb[inverse_indeces],
            f"t12_cos_{k}": t12[0][inverse_indeces], 
            f"t23_cos_{k}": t23[0][inverse_indeces], 
            f"t45_cos_{k}": t45[0][inverse_indeces], 
            f"t56_cos_{k}": t56[0][inverse_indeces], 
            f"t78_cos_{k}": t78[0][inverse_indeces], 
            f"t89_cos_{k}": t89[0][inverse_indeces], 
            f"t108_cos_{k}": t108[0][inverse_indeces], 
            f"t811_cos_{k}": t811[0][inverse_indeces], 
            f"t110_cos_{k}": t110[0][inverse_indeces],
            f"t311_cos_{k}": t311[0][inverse_indeces],
            f"t28_cos_{k}": t28[0][inverse_indeces],
            f"t47_cos_{k}": t47[0][inverse_indeces], 
            f"t69_cos_{k}": t69[0][inverse_indeces], 
            f't75_cos_{k}': t75[0][inverse_indeces], 
            f't95_cos_{k}': t95[0][inverse_indeces], 
            f't42_cos_{k}': t42[0][inverse_indeces], 
            f't62_cos_{k}': t62[0][inverse_indeces], 
            f't25_cos_{k}': t25[0][inverse_indeces],
            f't85_cos_{k}': t85[0][inverse_indeces], 
            f'dihedral_cos_{k}': dihedral_cos[inverse_indeces],
            f'dihedral_sin_{k}': dihedral_sin[inverse_indeces]
        }

        del t12, t23, t45, t56, t78, t89, t75, t95, t42, t62, t108, t811, t110, t311, t28, t47, t69, t25, t85, dihedral_cos, dihedral_sin
        gc.collect()
    return final_dict



def add_gmf_to_df(df, model_chains):
    '''
    adds geometrical features to the dataframe retrieved from pdb
    :param df: dataframe
    :param model_chains: list of chain ids
    :return: dataframe containing geometrical features
    '''
    DICTS = []
    for ch in model_chains:
        chain_df = df[df['chain']==ch]
        res_labels = chain_df['res_id'].to_numpy()
        unq_df = get_unique_df(chain_df, subset='residue', keep='first')
        sidechain_centroid_array = unq_df[['sidechain_pseudopos_x', 'sidechain_pseudopos_y', 'sidechain_pseudopos_z']].to_numpy()
        backbone_centroid_array = unq_df[['backbone_pseudopos_x', 'backbone_pseudopos_y', 'backbone_pseudopos_z']].to_numpy()
        print('################SIDE',sidechain_centroid_array.shape)
        dict = compute_geometrical_features(sidechain_centroid_array,
                                            backbone_centroid_array,
                                            res_labels)
        out_df = pd.DataFrame(dict)
        DICTS.append(out_df)
    DICTS = pd.concat(DICTS).reset_index(drop=True)
    DICTS = pd.concat([df, DICTS], axis=1)
    del dict, out_df, df
    gc.collect()
    return DICTS

def get_interactions_from_df(df, model_chains):
    df_res = get_unique_df(df, subset='residue', keep='first')
    arr1 = df_res[['sidechain_pseudopos_x', 'sidechain_pseudopos_y', 'sidechain_pseudopos_z']].to_numpy()
    lengths = []
    for ch in model_chains:
        lengths.append(len(df_res[df_res['chain']==ch]))
    interacting_pairs, non_interacting_pairs = find_inter_residues(arr1, lengths, inter_threshold=7)
    del df_res, arr1
    gc.collect()
    return interacting_pairs, non_interacting_pairs


def Extract_and_Save_from_PDB(input_file, from_dill=True, saving_dir='../database'):
    id_name = input_file.split('/')[-1].replace('.dill', '').replace('.pdb', '')
    print(f'#### 1- Extract for id {id_name}')
    os.makedirs(saving_dir, exist_ok=True)
    if from_dill:
        print(f'Dill State {id_name}')
        data = read_dill_and_get_dfs(input_file) # reads and cocats dfs from dill file
        pdb_file = saving_dir + '/' + id_name + '.pdb'
        df_to_pdb(data, default_cols=True, cols=None, filename=pdb_file) # saves pdb from dfs
        print(f'#### 2- DF to PDB done {pdb_file}')
    else:
        print(f'PDB State {id_name}')
        pdb_file = input_file
    structure = clean_and_renumber_pdb(pdb_file, pdb_file) # cleans and saves the structure
    print(f'#### 3- Structure cleaned and renumbered {id_name}')
    df, model_chains = pdb_to_df(pdb_file, structure, add_sasa=True, add_cx=True, pdb_id=id_name)
    print(f'#### 4- pdb to df done {id_name}')
    unique_labels, inverse_indeces = np.unique(np.array(df.residue.tolist()), return_inverse=True)
    hsaacs, UN, DN = get_hsaac_for_pdb_residues(structure, len(unique_labels))
    hsaacs , UN, DN = hsaacs[inverse_indeces], UN[inverse_indeces], DN[inverse_indeces]
    df['UN'], df['DN'] = UN, DN


    print(f'#### 5- hsaacs done {id_name}')
    df = add_gmf_to_df(df, model_chains)
    interacting_pairs, non_interacting_pairs = get_interactions_from_df(df, model_chains)
    print(f'#### 6- interacting_pairs done {id_name}')
    data = {'id_name':id_name, 'pdb_file':pdb_file, 'df':df, 'hsaacs':hsaacs,
            'interacting_pairs':interacting_pairs, 'non_interacting_pairs':non_interacting_pairs}
    with open(saving_dir + '/' + id_name + '.dill', 'wb') as f:
        dill.dump(data, f)
    print(f'#### 7- data saved {id_name}', saving_dir + '/' + id_name + '.dill')
    pdb_file, df, hsaacs, interacting_pairs, non_interacting_pairs, UN, DN = None, None, None, None, None, None, None
    del pdb_file, df, hsaacs, interacting_pairs, non_interacting_pairs, UN, DN
    gc.collect()
    return data



def extract_feature_from_dill(inputs, output, cols=None):

    ARR = []
    for i, input in enumerate(inputs):
        df = get_df_from_dill(input)
        array = load_data_from_df(df, cols)
        ARR.append(array)
    ARR = np.concatenate(ARR)
    np.save(output, ARR)
    print(input, '--->', output, '--- shape:', ARR.shape)
    return ARR

#profiler = cProfile.Profile()
#profiler.enable()
#df, model_chains = pdb_to_df(pdb_file='1A1N.pdb')
#df = add_gmf_to_df(df, model_chains)
#df.to_csv('../tmp/df.tsv', sep='\t', index=0)
#df = pd.read_csv('../raw/e9/tmp.tsv', sep='\t', header=0)
#Extract_and_Save_from_PDB('../raw/e9/2e9f.pdb1_0.dill', from_dill=True, saving_dir='../tmp_database/e9')
#print(df)