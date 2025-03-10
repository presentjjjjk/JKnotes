from rdkit import Chem
from rdkit.Chem import Draw

# Create a molecule from SMILES (ethylbenzene)
smiles = "CCc1ccccc1"
mol = Chem.MolFromSmiles(smiles)

# Define atom indices to highlight.
# For ethylbenzene, we choose the benzene ring atoms.
# Typically, for ethylbenzene the benzene ring is represented by indices 2 to 7.
highlight_atoms = [2, 3, 4, 5, 6, 7]

# Find bonds connecting these highlighted atoms.
highlight_bonds = []
for bond in mol.GetBonds():
    begin_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()
    # Check if both atoms in the bond are part of the highlight_atoms list.
    if begin_idx in highlight_atoms and end_idx in highlight_atoms:
        highlight_bonds.append(bond.GetIdx())

# Draw the molecule with the specified atoms and bonds highlighted.
img = Draw.MolToImage(mol,
                      size=(300, 300),
                      highlightAtoms=highlight_atoms,
                      highlightBonds=highlight_bonds)
img.save("highlighted_molecule.png")
print(highlight_bonds)