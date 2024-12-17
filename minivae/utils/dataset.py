"""
Function to obtain a Hugging Face Dataset of the data presented in the original scVAE paper, 
and openly shared in the PBMC dataset.

This script subsampled the original dataset, limiting the sequencing data to 2k cells per category
"""

import os
import tarfile
import scipy.io
import numpy as np
from datasets import Dataset, DatasetDict
import uuid

# Define the list of labels
LABELS = [
    "B_cells",
    "Hematopoietic_progenitor_cells",
    "Helper_T_cells",
    "Regulatory_T_cells",
    "Naive_Helper_T_cells",
    "Memory_Helper_T_cells",
    "Natural_Killer_cells",
    "Cytotoxic_T_cells",
    "Naive_Cytotoxic_T_cells"
]

def read_matrix_from_tar(tar_path):
    """
    Reads gene expression matrix and metadata from a tar.gz file.
    Returns the expression matrix, gene names, and cell barcodes.
    """
    matrix = None
    gene_names = None
    cell_names = None
    
    with tarfile.open(tar_path, mode="r:gz") as tarball:
        for member in tarball:
            if member.name.endswith("matrix.mtx"):
                with tarball.extractfile(member) as f:
                    matrix = scipy.io.mmread(f).T.tocsr()
            elif member.name.endswith("genes.tsv"):
                with tarball.extractfile(member) as f:
                    gene_names = np.array([
                        line.decode('utf-8').strip() 
                        for line in f.readlines()
                    ], dtype='U')
            elif member.name.endswith("barcodes.tsv"):
                with tarball.extractfile(member) as f:
                    cell_names = np.array([
                        line.decode('utf-8').strip() 
                        for line in f.readlines()
                    ], dtype='U')
    
    # Create dictionary matching scVAE format
    data_dictionary = {
        "values": matrix,
        "feature names": gene_names,
        "example names": cell_names,
    }

    if any([v is None for v in list(data_dictionary.values())]):
        raise ValueError("one of values, feature names or example names is still none!")
    
    return data_dictionary

def prepare_dataset(data, labels, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    """
    Creates a Hugging Face dataset from gene expression data and labels and splits into train/val/test.
    
    Args:
        data: List of gene expression data
        labels: List of corresponding labels
        train_size: Proportion of data for training.
        val_size: Proportion of data for validation.
        test_size: Proportion of data for testing.
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict containing train, validation, and test splits
    """
    # First create the full dataset
    full_dataset = Dataset.from_dict({
        "gene_expression": data,
        "label": labels
    })
    
    # Split the dataset into train, validation, and test
    splits = full_dataset.train_test_split(
        test_size=(val_size + test_size),
        seed=seed
    )
    
    # Further split the test portion into validation and test
    test_and_val = splits['test'].train_test_split(
        test_size=test_size/(test_size + val_size),
        seed=seed
    )
    
    # Create the final dataset dictionary
    final_dataset = {
        'train': splits['train'],
        'validation': test_and_val['train'],
        'test': test_and_val['test']
    }
    
    return final_dataset

def main(data_dir, push_to_hub="fracapuano/scRNA"):
    """
    Main function to create and push the Hugging Face dataset.
    Args:
        data_dir: Directory containing .tar.gz files
        push_to_hub: Repository name to push to HF Hub (e.g., 'username/dataset-name')
    """
    all_sequences = []
    all_labels = []
    all_ids = []
    
    for label in LABELS:
        print(f"Processing {label}...")
        tar_path = os.path.join(data_dir, f"{label}.tar.gz")
        
        # Read data matrix and metadata
        data_dict = read_matrix_from_tar(tar_path)
        
        # Convert sparse matrix to dense array for each cell sequencing
        cell_expressions = data_dict["values"].toarray()
        
        # Take only the first 2000 cells (subsampling the dataset)
        cell_expressions = cell_expressions[:2000]
        
        # Generate unique IDs for selected cells
        unique_ids = [str(uuid.uuid4()) for _ in range(cell_expressions.shape[0])]
        
        # Append to lists
        all_sequences.extend(cell_expressions.tolist())
        all_labels.extend([label] * len(cell_expressions))
        all_ids.extend(unique_ids)
        
        # Clear memory
        del data_dict, cell_expressions
        
    # Create complete dataset dictionary
    dataset_dict = {
        'id': all_ids,
        'sequence': all_sequences,
        'label': all_labels
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train/val/test
    split_dataset = prepare_dataset(
        data=dataset['sequence'],
        labels=dataset['label'],
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        seed=42
    )
    
    # Convert to DatasetDict
    final_dataset = DatasetDict(split_dataset)
    
    # Push complete dataset to hub
    final_dataset.push_to_hub(
        push_to_hub,
    )
    
    print(f"Complete dataset (with splits) pushed to HuggingFace Hub")

if __name__ == "__main__":
    main(
        data_dir="../../data/original",
        push_to_hub="fracapuano/scRNA-2k"
    )