import torch.nn as nn
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import torch.serialization
from torch_geometric.datasets import QM9

# 定义一个支持批处理的GNN层
class simple_gnn_layer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        self.activation = nn.ReLU()
    
    def message_passing(self,h,adj):
        '''改进后的消息传递，考虑边权重'''
        # 对邻接矩阵进行归一化处理
        row_sum = adj.sum(dim=2, keepdim=True) + 1e-6
        norm_adj = adj / row_sum  # 按行归一化
        
        h_prime = torch.bmm(norm_adj, h)  # 使用带权重的邻接矩阵
        return h_prime
    
    def update(self,h_prime,h):
        combined  = h + h_prime
        combined = self.linear(combined)
        return self.activation(combined)
    
    def forward(self,h,adj):
        h_prime = self.message_passing(h,adj)
        return self.update(h_prime,h)

class simple_gnn(nn.Module):
    def __init__(self,model_config_gnn,model_config_mlp):
        super().__init__()
        self.layers = nn.ModuleList([simple_gnn_layer(input_dim, output_dim) for input_dim, output_dim in model_config_gnn])
        mlp_layers = []
        for i in range(len(model_config_mlp)):
            if i == len(model_config_mlp)-1:
                mlp_layers.append(nn.Linear(model_config_mlp[i][0],model_config_mlp[i][1]))
            else:
                mlp_layers.append(nn.Linear(model_config_mlp[i][0],model_config_mlp[i][1]))
                mlp_layers.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_layers)
    
    # 修改全局池化方法
    def global_pooling(self,h):
        # h形状: [batch_size, num_nodes, features]
        pooled = torch.mean(h, dim=1)
        return pooled  # 添加非线性变换
    
    def mlp(self, h):
        # h形状: [batch_size, features]
        return self.mlp_layers(h)
    
    def forward(self,h,adj):
        # h形状: [batch_size, num_nodes, input_dim]
        # adj形状: [batch_size, num_nodes, num_nodes]
        for layer in self.layers:
            h = layer(h,adj)
        h = self.global_pooling(h)  # 输出形状 [batch_size, features]
        return self.mlp(h)

# 1. 加载数据集
dataset = QM9(root='dataset/QM9')[:40000]
import os
print("当前QM9数据集路径:", os.path.abspath('dataset/QM9'))
print(f"数据集大小: {len(dataset)} 个图")
print(f"节点特征维度: {dataset.num_node_features}")
print(f"边特征维度: {dataset.num_edge_features}")


# 2. 数据预处理和划分
def process_data(data):
    # 节点特征标准化
    x = (data.x.float() - feature_mean) / feature_std
    
    # 获取原子坐标（单位：埃）
    atom_pos = data.pos  # 形状: [num_nodes, 3]
    
    # 计算真实的键长（欧氏距离）
    edge_index = data.edge_index
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    # 计算原子间距离（单位：埃）
    edge_length = torch.norm(atom_pos[src_nodes] - atom_pos[dst_nodes], dim=1)
    
    # 创建带权重的邻接矩阵（使用真实键长）
    num_nodes = x.size(0)
    adj = torch.zeros(num_nodes, num_nodes)
    
    # 填充邻接矩阵（考虑无向图的双向关系）
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        # 使用1/(键长+epsilon)作为权重（键长越短权重越大）
        weight = 1 / (edge_length[i] + 1e-6)
        adj[src, dst] = weight
        adj[dst, src] = weight  # 无向图需要双向赋值

    # 直接使用原始标签
    y = data.y[:, 0].float()
    
    return x, adj, y

# 划分数据集
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# 仅保留特征统计量计算
train_features = torch.cat([data.x.float() for data in train_dataset], dim=0)
feature_mean = train_features.mean(dim=0)
feature_std = train_features.std(dim=0) + 1e-6

# 2. 数据预处理和划分
def collate_fn(batch):
    """
    将一批图数据处理成批量格式
    """
    x_batch = []
    adj_batch = []
    y_batch = []
    
    for data in batch:
        x, adj, y = process_data(data)
        x_batch.append(x)
        adj_batch.append(adj)
        y_batch.append(y)
    
    # 添加填充和转换张量的步骤
    max_nodes = max([x.shape[0] for x in x_batch]) # x: [num_nodes, features]
    x_padded = [torch.nn.functional.pad(x, (0,0,0,max_nodes-x.shape[0])) for x in x_batch]
    adj_padded = [torch.nn.functional.pad(adj, (0,max_nodes-adj.shape[1],0,max_nodes-adj.shape[0])) for adj in adj_batch]
    
    return torch.stack(x_padded).to(device), torch.stack(adj_padded).to(device), torch.stack(y_batch).to(device)

# 3. 模型配置和初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU is available")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
else:
    print("using CPU")

# 创建数据加载器
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) #在这里应用刚刚定义的打包函数
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


model_config_gnn = [
    (11, 128),          # 增大初始维度
    (128, 128),         # 保持维度稳定
    (128, 128),         # 增加层数
    (128, 64)           # 逐步降维
]
model_config_mlp = [
    (64, 128),          # 减少MLP层数
    (128, 64),
    (64, 1)             # 直接输出
]

model = simple_gnn(model_config_gnn, model_config_mlp).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,           # 学习率需要降低1-2个数量级
    weight_decay=1e-6
)

# 训练配置
num_epochs = 100        # 增加训练轮次
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # 学习率调度
    optimizer, mode='min', factor=0.5, patience=5
)

# 4. 训练函数
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for x_batch, adj_batch, y_batch in train_loader:  
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(x_batch, adj_batch)  
        pred = pred.view(-1)
        y_batch = y_batch.view(-1)
        
        # 计算损失
        loss = criterion(pred, y_batch)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 5. 评估函数
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    total_mae = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, adj_batch, y_batch in data_loader:
            pred = model(x_batch, adj_batch)
            pred = pred.view(-1)
            y_batch = y_batch.view(-1)
            
            # 直接计算原始量纲指标
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
            
            mae = torch.mean(torch.abs(pred - y_batch))
            total_mae += mae.item()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    # 计算原始量纲的R²
    r2 = 1 - np.sum((np.array(all_targets) - np.array(all_preds)) ** 2) / np.sum((np.array(all_targets) - np.mean(all_targets)) ** 2)
    
    return total_loss / len(data_loader), total_mae / len(data_loader), r2


train_losses = []
train_maes = []
train_r2s = []
test_losses = []
test_maes = []
test_r2s = []

print("开始训练...")
for epoch in range(num_epochs):
    # 训练
    train_loss = train(model, train_loader, optimizer, criterion)
    
    # 评估训练集和测试集性能
    train_loss, train_mae, train_r2 = evaluate(model, train_loader)
    test_loss, test_mae, test_r2 = evaluate(model, test_loader)
    
    # 记录指标
    train_losses.append(train_loss)
    train_maes.append(train_mae)
    train_r2s.append(train_r2)
    test_losses.append(test_loss)
    test_maes.append(test_mae)
    test_r2s.append(test_r2)
    
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}')
        print(f'Test  - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}')
        print('-' * 50)

# 7. 最终结果
print(f'最终结果:')
print(f'Train - Loss: {train_losses[-1]:.4f}, MAE: {train_maes[-1]:.4f}, R²: {train_r2s[-1]:.4f}')
print(f'Test  - Loss: {test_losses[-1]:.4f}, MAE: {test_maes[-1]:.4f}, R²: {test_r2s[-1]:.4f}')

# 8. 绘制训练和测试曲线
plt.figure(figsize=(15, 5))

# 创建三个子图
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_maes, label='Training MAE')
plt.plot(test_maes, label='Test MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE Curves')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_r2s, label='Training R²')
plt.plot(test_r2s, label='Test R²')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('R² Curves')
plt.legend()

plt.tight_layout()
plt.savefig('result.png')

