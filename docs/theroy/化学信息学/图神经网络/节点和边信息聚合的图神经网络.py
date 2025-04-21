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
from torch_geometric.nn import MessagePassing, global_mean_pool
from sklearn.metrics import mean_absolute_error, r2_score

class GNNLayer(MessagePassing):
    def __init__(self, node_input_dim, node_output_dim, edge_input_dim, edge_output_dim, dropout=0.1):
        super().__init__(aggr="add")  # 使用加法聚合

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_input_dim * 2 + edge_input_dim, edge_output_dim),
            nn.BatchNorm1d(edge_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim + edge_output_dim, node_output_dim),
            nn.BatchNorm1d(node_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    
    def forward(self, x, edge_index, edge_attr):
        # 先更新边
        new_edge_attr = self.update_edges(x, edge_index, edge_attr)
        # 再更新节点
        # propagate是pyg的高度优化更新方法
        # 其会自动调用massage方法生成消息,然后根据aggr参数聚合消息,最后用update方法更新节点特征
        new_x = self.propagate(edge_index, x=x, edge_attr=new_edge_attr)
        return new_x, new_edge_attr
    
    def update_edges(self, x, edge_index, edge_attr):
        src, dst = edge_index
        src_features = x[src]
        dst_features = x[dst]
        edge_features = torch.cat([src_features, dst_features, edge_attr], dim=1)
        return self.edge_mlp(edge_features)
    
    def message(self, x_j, edge_attr):
        # 组合节点特征和边特征
        return torch.cat([x_j, edge_attr], dim=1)
    
    def update(self, aggr_out, x):
        # 更新节点特征
        return self.node_mlp(aggr_out)

class GNN(nn.Module):
    def __init__(self, model_config_gnn, model_config_mlp, edge_dim=None, dropout=0.05):
        super().__init__()
        
        # 添加节点预处理MLP层
        node_input_dim = model_config_gnn[0][0]
        self.node_preprocessing = nn.Sequential(
            nn.Linear(node_input_dim, node_input_dim*2),
            nn.BatchNorm1d(node_input_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_input_dim*2, node_input_dim),
            nn.BatchNorm1d(node_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 添加边预处理MLP层
        edge_input_dim = model_config_gnn[0][2]
        self.edge_preprocessing = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim*2),
            nn.BatchNorm1d(edge_input_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_input_dim*2, edge_input_dim),
            nn.BatchNorm1d(edge_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 初始化GNN层
        self.layers = nn.ModuleList([
            GNNLayer(node_input_dim, node_output_dim, edge_input_dim, edge_output_dim, dropout) 
            for (node_input_dim, node_output_dim, edge_input_dim, edge_output_dim) in model_config_gnn
        ])
               
        mlp_layers = []
        for i in range(len(model_config_mlp) - 1):
            mlp_layers.append(nn.Linear(model_config_mlp[i][0], model_config_mlp[i][1]))
            if i < len(model_config_mlp) - 2:  # 除了最后一层外都添加BatchNorm
                mlp_layers.append(nn.BatchNorm1d(model_config_mlp[i][1]))
            
            # 倒数第二层使用tanh激活函数，其他层使用ReLU
            if i == len(model_config_mlp) - 3:  # 倒数第二层
                mlp_layers.append(nn.Tanh())
            else:
                mlp_layers.append(nn.ReLU())
                
            if i < len(model_config_mlp) - 2:  # 除了最后一层外都添加Dropout
                mlp_layers.append(nn.Dropout(dropout))
        # 添加最后一层（没有激活函数）
        mlp_layers.append(nn.Linear(model_config_mlp[-2][1], model_config_mlp[-1][1]))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
    # 全局池化
    def node_global_pooling(self, h):
        """
        对节点特征进行全局池化
        h: [total_nodes, node_dim]  # 实际形状
        return: [node_dim]  # 实际返回形状
        """
        return torch.mean(h, dim=0)  # 沿节点维度平均
    
    def edge_global_pooling(self, edge_attr):
        """
        对边特征进行全局池化
        edge_attr: [total_edges, edge_dim]  # 实际形状
        return: [edge_dim]  # 实际返回形状
        """
        return torch.mean(edge_attr, dim=0)  # 沿边维度平均

    def forward(self, h, edge_index, edge_attr, batch):
        """
        h: 节点特征 [total_nodes, node_dim]
        edge_index: 边索引 [2, total_edges]
        edge_attr: 边特征 [total_edges, edge_dim]
        batch: 批次索引 [total_nodes]
        """
        # 先应用预处理MLP层
        h = self.node_preprocessing(h)
        edge_attr = self.edge_preprocessing(edge_attr)
        
        # 通过所有GNN层
        for layer in self.layers:
            h, edge_attr = layer(h, edge_index, edge_attr)
        
        # 使用PyG提供的全局池化函数
        pooled_nodes = global_mean_pool(h, batch)  # [batch_size, node_dim]
        
        # 为边创建批次索引(基于它们连接的节点)
        edge_batch = batch[edge_index[0]]
        pooled_edges = global_mean_pool(edge_attr, edge_batch)  # [batch_size, edge_dim]
        
        # 拼接特征
        combined_features = torch.cat([pooled_nodes, pooled_edges], dim=1)
        
        # 应用MLP生成预测
        predictions = self.mlp(combined_features)
        
        # 确保输出形状为 [batch_size, 1]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        
        return predictions
        

# 1. 加载数据集
dataset = QM9(root='dataset/QM9')[:10000]
print(f"数据集大小: {len(dataset)} 个图")
print(f"节点特征维度: {dataset.num_node_features}")
print(f"边特征维度: {dataset.num_edge_features}")

# 2. 数据预处理和划分
def process_data(data):
    # 节点特征标准化
    x = (data.x.float() - node_feature_mean) / node_feature_std
        
    # 获取边索引
    edge_index = data.edge_index
    
    # 获取并标准化边特征
    edge_attr = (data.edge_attr.float() - edge_feature_mean) / edge_feature_std
    
    # 标签归一化
    raw_y = data.y[:, 7].float() - data.y[:, 6].float()
    y = (raw_y - label_mean) / label_std
    
    return x, edge_index, edge_attr, y

# 划分数据集（添加验证集）
train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, random_state=42)

print(f"训练集大小: {len(train_dataset)} 个图")
print(f"验证集大小: {len(val_dataset)} 个图")
print(f"测试集大小: {len(test_dataset)} 个图")

# 计算特征统计量
train_node_features = torch.cat([data.x.float() for data in train_dataset], dim=0)
node_feature_mean = train_node_features.mean(dim=0)
node_feature_std = train_node_features.std(dim=0) + 1e-6

train_edge_features = torch.cat([data.edge_attr.float() for data in train_dataset], dim=0)
edge_feature_mean = train_edge_features.mean(dim=0)
edge_feature_std = train_edge_features.std(dim=0) + 1e-6

# 计算标签统计量用于归一化
train_labels = torch.tensor([data.y[0, 7].float() - data.y[0, 6].float() for data in train_dataset])
label_mean = train_labels.mean()
label_std = train_labels.std() + 1e-6
print(f"标签均值: {label_mean.item():.4f}, 标签标准差: {label_std.item():.4f}")

def collate_fn(batch):
    """
    将一批图数据处理成批量格式,包含节点特征、边索引、边特征和batch索引
    """
    x_batch = []
    edge_index_batch = []
    edge_attr_batch = []
    y_batch = []
    batch_indices = []  # 新增：记录每个节点属于哪个图
    
    batch_size = len(batch)
    cumulative_nodes = 0  # 用于调整边索引
    
    for i, data in enumerate(batch):
        x, edge_index, edge_attr, y = process_data(data)
        num_nodes = x.size(0)
        
        # 存储处理后的数据
        x_batch.append(x)
        
        # 调整边索引以考虑批处理
        edge_index_adjusted = edge_index.clone()
        edge_index_adjusted += cumulative_nodes
        edge_index_batch.append(edge_index_adjusted)
        
        edge_attr_batch.append(edge_attr)
        y_batch.append(y)
        
        # 为这个图的所有节点添加批次索引
        batch_indices.append(torch.full((num_nodes,), i, dtype=torch.long))
        
        cumulative_nodes += num_nodes  # 更新累积节点数
    
    # 组合所有图的数据
    x_combined = torch.cat(x_batch, dim=0)
    edge_index_combined = torch.cat(edge_index_batch, dim=1)
    edge_attr_combined = torch.cat(edge_attr_batch, dim=0)
    y_combined = torch.stack(y_batch)
    batch_combined = torch.cat(batch_indices, dim=0)  # 新增：合并batch索引
    
    return x_combined.to(device), edge_index_combined.to(device), edge_attr_combined.to(device), y_combined.to(device), batch_combined.to(device)

# 3. 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU可用")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
else:
    print("使用CPU")

# 创建数据加载器
batch_size = 128  # 因为边信息需要更多内存，批量大小适当减小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 4. 模型配置和初始化
# 配置GNN层: (节点输入维度, 节点输出维度, 边输入维度, 边输出维度)
model_config_gnn = [
    (dataset.num_node_features, 128, dataset.num_edge_features, 64),  
    (128, 128, 64, 64),                                              
    (128, 64, 64, 32),
    (64, 64, 32,32),
    (64, 64, 32, 32)                                                
]

# 配置MLP层: (输入维度, 输出维度)
model_config_mlp = [
    (64 + 32, 64),  
    (64, 32),
    (32,16),
    (16,1)       
]

# 增加模型参数，包括dropout
model = GNN(model_config_gnn, model_config_mlp, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4  # 增强L2正则化
)

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=7, 
    min_lr=1e-6
)

# 添加评估指标跟踪
from sklearn.metrics import mean_absolute_error, r2_score

# 训练和评估函数
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, edge_index, edge_attr, y, batch in loader:
        optimizer.zero_grad()
        pred = model(x, edge_index, edge_attr, batch)
        loss = criterion(pred.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, edge_index, edge_attr, y, batch in loader:
            pred = model(x, edge_index, edge_attr, batch)
            loss = criterion(pred.view(-1), y.view(-1))
            total_loss += loss.item() * y.size(0)
            
            # 反归一化预测值和标签用于计算真实指标
            pred_denorm = pred.view(-1) * label_std + label_mean
            y_denorm = y.view(-1) * label_std + label_mean
            
            all_preds.append(pred_denorm.cpu().numpy())
            all_labels.append(y_denorm.cpu().numpy())
    
    # 计算评估指标
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_labels = np.concatenate([l.flatten() for l in all_labels])
    
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    return total_loss / len(loader.dataset), mae, r2

# 初始化指标跟踪列表
train_losses, train_maes, train_r2s = [], [], []
test_losses, test_maes, test_r2s = [], [], []

# 训练配置
num_epochs = 200  # 增加最大epoch数量
patience = 15  # 连续patience个epoch没有改进则停止
best_val_loss = float('inf')
early_stop_counter = 0
best_model_state = None

print("开始训练...")
for epoch in range(num_epochs):
    # 训练
    train_loss = train(model, train_loader, optimizer, criterion)
    
    # 评估训练集、验证集和测试集性能
    train_loss, train_mae, train_r2 = evaluate(model, train_loader)
    val_loss, val_mae, val_r2 = evaluate(model, val_loader)
    test_loss, test_mae, test_r2 = evaluate(model, test_loader)
    
    # 记录指标
    train_losses.append(train_loss)
    train_maes.append(train_mae)
    train_r2s.append(train_r2)
    test_losses.append(test_loss)
    test_maes.append(test_mae)
    test_r2s.append(test_r2)
    
    # 学习率调整（基于验证集损失）
    scheduler.step(val_loss)
    # 如果需要显示当前学习率，可以使用 get_last_lr()
    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Learning Rate: {current_lr:.6f}')
        print(f'Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}')
        print(f'Test  - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}')
        print('-' * 50)
    
    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        early_stop_counter += 1
    
    # 检查是否应该早停
    if early_stop_counter >= patience:
        print(f'早停: 验证集损失在{patience}个epoch内没有改善。')
        break

# 加载最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("已加载最佳模型状态")

# 最终评估最佳模型
final_train_loss, final_train_mae, final_train_r2 = evaluate(model, train_loader)
final_val_loss, final_val_mae, final_val_r2 = evaluate(model, val_loader)
final_test_loss, final_test_mae, final_test_r2 = evaluate(model, test_loader)

# 8. 最终结果
print(f'最终结果:')
print(f'训练集 - Loss: {final_train_loss:.4f}, MAE: {final_train_mae:.4f}, R²: {final_train_r2:.4f}')
print(f'验证集 - Loss: {final_val_loss:.4f}, MAE: {final_val_mae:.4f}, R²: {final_val_r2:.4f}')
print(f'测试集 - Loss: {final_test_loss:.4f}, MAE: {final_test_mae:.4f}, R²: {final_test_r2:.4f}')

# 9. Plot training and testing curves
plt.figure(figsize=(15, 5))

# Create three subplots
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
plt.savefig('edge_aggregation_results.png')
        

