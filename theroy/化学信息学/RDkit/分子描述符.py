from rdkit.Chem import Descriptors
from rdkit import Chem

# 列出所有描述符
descriptors = []
for name, function in Descriptors._descList:
    descriptors.append(name)
print(f"总共有 {len(descriptors)} 个描述符")
print("前10个描述符示例：")
for desc in descriptors[:10]:
    print(desc)

from rdkit.Chem import Crippen,Descriptors

def get_common_descriptors(mol):
    return {
        'MW': Descriptors.ExactMolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'Rotatable_Bonds': Descriptors.NumRotatableBonds(mol),
        'Rings': Descriptors.RingCount(mol),
        'Aromatic_Rings': Descriptors.NumAromaticRings(mol)
    }

# 使用示例
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
common_desc = get_common_descriptors(mol)
for name, value in common_desc.items():
    print(f"{name}: {value}")