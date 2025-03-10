from rdkit import Chem
from rdkit.Chem import AllChem

# Example complex molecule containing a benzene ring.
# Replace this with your molecule's SMILES.
smiles = "c1ccccc1CC(=O)O"  # for instance, a benzoic acid derivative
mol = Chem.MolFromSmiles(smiles)
Chem.SanitizeMol(mol)   # 检查分子是否合法
# Define a benzene ring pattern (SMARTS)
benzene_pattern = Chem.MolFromSmarts("c1ccccc1")

# Find benzene substructure matches in the molecule
matches = mol.GetSubstructMatches(benzene_pattern)
print(matches)
if matches:
    # Choose the first match; this returns a tuple of atom indices (one per ring atom)
    match = matches[0]
    print(match)
    # Replace one carbon with nitrogen.
    # Here, we choose the first atom in the match.
    atom_to_replace = match[1]
    mol.GetAtomWithIdx(atom_to_replace).SetAtomicNum(14)  # 7 is nitrogen

    # Resanitize the molecule to update aromaticity and valence
    Chem.SanitizeMol(mol)
    new_smiles_direct = Chem.MolToSmiles(mol)
    print("Modified molecule (atom replacement):", new_smiles_direct)
else:
    print("No benzene ring found using the SMARTS pattern.")

