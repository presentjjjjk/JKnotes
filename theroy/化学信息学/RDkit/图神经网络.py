import dgl
import tensorflow as tf
from rdkit import Chem
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 分子图转换函数
def smiles_to_graph(smiles):
    """将SMILES转换为DGL图"""
    mol = Chem.MolFromSmiles(smiles)
    
    # 获取原子特征
    def get_atom_features(atom):
        return tf.convert_to_tensor([
            atom.GetAtomicNum(),          # 原子序数
            atom.GetDegree(),             # 度
            atom.GetFormalCharge(),       # 形式电荷
            atom.GetNumRadicalElectrons(),# 自由基电子数
            int(atom.GetIsAromatic()),    # 是否芳香
            atom.GetHybridization().real, # 杂化类型
        ], dtype=tf.float32)
    
    # 获取边特征
    def get_bond_features(bond):
        return tf.convert_to_tensor([
            bond.GetBondTypeAsDouble(),   # 键类型
            int(bond.GetIsConjugated()),  # 是否共轭
            int(bond.IsInRing()),         # 是否在环中
        ], dtype=tf.float32)
    
    # 构建图
    g = dgl.graph([])
    
    # 添加节点
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    node_features = []
    for i in range(num_atoms):
        node_features.append(get_atom_features(mol.GetAtomWithIdx(i)))
    g.ndata['h'] = tf.stack(node_features)
    
    # 添加边
    src_list = []
    dst_list = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        src_list.extend([i, j])
        dst_list.extend([j, i])
        edge_features.extend([get_bond_features(bond), get_bond_features(bond)])
    
    g.add_edges(src_list, dst_list)
    g.edata['h'] = tf.stack(edge_features)
    
    return g

# 2. 图卷积层
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, out_feats):
        super(GCNLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(out_feats)
        
    def call(self, g, features):
        with g.local_scope():
            g.ndata['h'] = features
            # 消息传递
            g.update_all(
                message_func=dgl.function.copy_u('h', 'm'),
                reduce_func=dgl.function.mean('m', 'h_neigh')
            )
            # 更新节点特征
            h_neigh = g.ndata['h_neigh']
            h = self.dense(h_neigh)
            return tf.nn.relu(h)

# 3. GCN模型
class GCN(tf.keras.Model):
    def __init__(self, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(hidden_size)
        self.gcn2 = GCNLayer(hidden_size)
        self.predict = tf.keras.layers.Dense(out_feats)
        
    def call(self, g):
        h = g.ndata['h']
        h = self.gcn1(g, h)
        h = self.gcn2(g, h)
        # 全局平均池化
        g.ndata['h'] = h
        h_g = dgl.mean_nodes(g, 'h')
        return self.predict(h_g)

# 4. 数据生成器
class MoleculeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, smiles_list, labels, batch_size=32):
        self.smiles_list = smiles_list
        self.labels = labels
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.smiles_list) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_smiles = self.smiles_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_graphs = [smiles_to_graph(smiles) for smiles in batch_smiles]
        return dgl.batch(batch_graphs), tf.convert_to_tensor(batch_labels, dtype=tf.float32)

# 5. 主程序
def main():
    # 示例数据
    smiles_list = [
        'CCO',      # 乙醇
        'CCCO',     # 正丙醇
        'CCCCO',    # 正丁醇
        'CC(=O)O',  # 乙酸
        'CCN',      # 乙胺
    ]
    
    # 虚构的溶解度数据
    solubility = np.array([
        [1000.0],  # 乙醇
        [800.0],   # 正丙醇
        [600.0],   # 正丁醇
        [1200.0],  # 乙酸
        [900.0],   # 乙胺
    ], dtype=np.float32)
    
    # 创建数据生成器
    data_gen = MoleculeDataGenerator(smiles_list, solubility, batch_size=2)
    
    # 构建模型
    model = GCN(
        hidden_size=32,  # 隐藏层维度
        out_feats=1      # 输出维度（溶解度）
    )
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # 训练模型
    history = model.fit(
        data_gen,
        epochs=100,
        verbose=1
    )
    
    # 测试预测
    test_smiles = 'CCCCCCO'  # 正己醇
    test_graph = smiles_to_graph(test_smiles)
    test_batch = dgl.batch([test_graph])
    pred = model.predict(test_batch)
    print(f"\n预测{test_smiles}的溶解度为: {pred[0][0]:.2f} mg/L")
    
    # 绘制训练过程
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.title('模型训练过程')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()