import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

# 超参数配置
config = {
    "seq_len": 100,        # 输入序列长度
    "input_dim": 1,        # 输入特征维度
    "output_dim": 1,       # 输出特征维度
    "d_model": 256,        # Transformer特征维度
    "nhead": 4,            # 注意力头数
    "num_layers": 4,       # Transformer层数
    "dropout": 0.1,        # Dropout概率
    "batch_size": 32,      # 批大小
    "lr": 1e-4,            # 学习率
    "num_epochs": 1000,     # 训练轮数
    "kernel_size": 3       # 因果卷积核大小
}

# 1. 因果数据生成
class CausalConvDataset(Dataset):
    def __init__(self, num_samples=1000):
        # 生成随机输入信号 (batch, seq_len, input_dim)
        self.X = torch.randn(num_samples, config["seq_len"], config["input_dim"])
        
        # 创建因果卷积核
        self.kernel = torch.tensor([0,0,0,0,0,1,0])
        
        # 应用因果卷积
        self.Y = torch.zeros_like(self.X)
        for i in range(num_samples):
            # 因果卷积实现
            x = self.X[i].squeeze(-1)  # (seq_len,)
            y = torch.zeros_like(x)
            for t in range(config["seq_len"]):
                start = max(0, t - config["kernel_size"] + 1)
                y[t] = (x[start:t+1] * self.kernel[-(t-start+1):]).sum()
            self.Y[i] = y.unsqueeze(-1)
        
        # 数据标准化
        self.X_mean, self.X_std = self.X.mean(), self.X.std()
        self.Y_mean, self.Y_std = self.Y.mean(), self.Y.std()
        self.X = (self.X - self.X_mean) / self.X_std
        self.Y = (self.Y - self.Y_mean) / self.Y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 2. 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# 3. 因果Transformer模型
class CausalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(config["input_dim"], config["d_model"])
        self.pos_encoder = PositionalEncoding(config["d_model"])
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config["d_model"],
                nhead=config["nhead"],
                dim_feedforward=4*config["d_model"],
                dropout=config["dropout"]
            ),
            num_layers=config["num_layers"]
        )
        self.output_layer = nn.Linear(config["d_model"], config["output_dim"])
        
        # 因果掩码
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(config["seq_len"], config["seq_len"]), diagonal=1).bool()
        )

    def forward(self, x):
        # x形状: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        
        # Transformer编码器（应用因果掩码）
        x = self.transformer(x, mask=self.causal_mask)  # (seq_len, batch, d_model)
        
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        return self.output_layer(x)  # (batch, seq_len, output_dim)

# 4. 训练准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集和数据加载器
train_dataset = CausalConvDataset(num_samples=1000)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

# 模型初始化
model = CausalTransformer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# 5. 训练循环
for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0
    
    for batch_idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

# 6. 推理示例
model.eval()
with torch.no_grad():
    test_sample = train_dataset[0][0].unsqueeze(0).to(device)  # (1, seq_len, input_dim)
    prediction = model(test_sample)  # (1, seq_len, output_dim)
    print("Prediction shape:", prediction.shape)