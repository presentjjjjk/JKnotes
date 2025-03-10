import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_morgan_fingerprint(smiles, radius=2, nBits=2048):
    # Initialize a list to count substructure occurrences
    substructCnt = {}
    
    # Loop over all possible substructures of length 1 to radius
    for i in range(1, radius + 1):
        for j in range(len(smiles) - i + 1):
            substruct = smiles[j:j + i]
            
            if substruct in substructCnt:
                substructCnt[substruct] += 1
            else:
                substructCnt[substruct] = 1
    
    # Sort the substructures
    sortedSubstruct = sorted(substructCnt.keys())
    print(sortedSubstruct)
    # Initialize the fingerprint
    fingerprint = []
    
    # For each substructure, convert the count to a 16-bit binary representation
    for substruct in sortedSubstruct:
        substructBinary = [int(bit) for bit in bin(substructCnt[substruct])[2:].zfill(16)]
        fingerprint.extend(substructBinary)
    
    # Ensure the fingerprint has the correct number of bits
    if len(fingerprint) >= nBits:
        fingerprint = fingerprint[:nBits]
    else:
        fingerprint.extend([0] * (nBits - len(fingerprint)))
    
    return np.array(fingerprint)


def generate_rdkit_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol=Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

smiles_1 = 'CCO'
smiles_2 = 'CCF'


print(generate_morgan_fingerprint(smiles_1,radius=2,nBits=128))
print(generate_morgan_fingerprint(smiles_2,radius=2,nBits=128))
print(generate_rdkit_morgan_fingerprint(smiles_1,radius=2,nBits=128))
print(generate_rdkit_morgan_fingerprint(smiles_2,radius=2,nBits=128))

smiles = "CC(C)c4nc(CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](Cc1ccccc1)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc3cncs3)cs4"  # Example SMILES string
mol = Chem.MolFromSmiles(smiles)
from rdkit.Chem import Draw

# 创建高清图像
img = Draw.MolToImage(mol, 
                     size=(800,600),  # 更大的尺寸
                     dpi=1000,  # 更高的DPI
                     )

# 如果要保存高清图像
img.save('molecule_hd.png', dpi=(300,300))