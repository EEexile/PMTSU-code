import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 添加当前目录到 Python 路径（确保能导入 data.py 等）
sys.path.append(str(Path(__file__).parent))

from data import load_ssl_features, train_valid_test_iemocap_dataloader

# --- 模型定义（必须与训练时完全一致）---
import torch.nn as nn
from torch.nn import MultiheadAttention

class CrossTaskAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emo_query = nn.Linear(hidden_dim, hidden_dim)
        self.vad_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.attn = MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    
    def forward(self, emo_feat, vad_feats):
        Q = self.emo_query(emo_feat).unsqueeze(1)
        KV = self.vad_proj(vad_feats)
        K, V = torch.split(KV, self.hidden_dim, dim=-1)
        attn_out, attn_weights = self.attn(Q, K, V, need_weights=True)
        return attn_out, attn_weights

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, attn_out, emo_feat):
        attn_out = attn_out.squeeze(1)
        gate_input = torch.cat([emo_feat, attn_out], dim=-1)
        update_gate = self.update_gate(gate_input)
        fused = emo_feat + update_gate * (attn_out - emo_feat)
        return self.transform(fused), update_gate

class BaseModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4, num_heads=4):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.emo_adapter = nn.Linear(hidden_dim * 2, hidden_dim)
        self.vad_adapters = nn.ModuleDict({
            'valence': nn.Linear(hidden_dim * 2, hidden_dim),
            'arousal': nn.Linear(hidden_dim * 2, hidden_dim),
            'dominance': nn.Linear(hidden_dim * 2, hidden_dim)
        })
        self.valence_head = nn.Linear(hidden_dim, 3)
        self.arousal_head = nn.Linear(hidden_dim, 3)
        self.dominance_head = nn.Linear(hidden_dim, 3)
        self.cross_attention = CrossTaskAttention(hidden_dim, num_heads)
        self.fusion = GatedFusion(hidden_dim)
        self.emotion_head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, padding_mask=None):
        shared_features = self.shared_layer(x)
        emo_feat = self._get_emo_feature(shared_features, padding_mask)
        vad_feats, valence_out, arousal_out, dominance_out = self._get_vad_features(shared_features, padding_mask)
        attn_out, attn_weights = self.cross_attention(emo_feat, vad_feats)
        fused_feature, fusion_weights = self.fusion(attn_out, emo_feat)
        fused_feature = fused_feature + emo_feat
        emotion_out = self.emotion_head(fused_feature)
        return emotion_out, valence_out, arousal_out, dominance_out, attn_weights, fusion_weights

    def _get_emo_feature(self, features, mask):
        emo_feat = self.emo_adapter(features)
        emo_feat = nn.functional.relu(emo_feat)
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            valid = emo_feat * (1 - mask_float)
            emo_feat = valid.sum(dim=1) / (1 - mask_float).sum(dim=1)
        else:
            emo_feat = emo_feat.mean(dim=1)
        return emo_feat 

    def _get_vad_features(self, features, mask):
        vad_feats = []
        outputs = []
        for task, adapter in self.vad_adapters.items():
            task_feat = adapter(features)
            task_feat = nn.functional.relu(task_feat)
            if mask is not None:
                mask_float = mask.unsqueeze(-1).float()
                valid_feat = task_feat * (1 - mask_float)
                avg_feat = valid_feat.sum(dim=1) / (1 - mask_float).sum(dim=1)
            else:
                avg_feat = task_feat.mean(dim=1)
            vad_feats.append(avg_feat)
            if task == 'valence':
                outputs.append(self.valence_head(avg_feat))
            elif task == 'arousal':
                outputs.append(self.arousal_head(avg_feat))
            else:
                outputs.append(self.dominance_head(avg_feat))
        return torch.stack(vad_feats, dim=1), *outputs


# --- 主函数 ---
def main():
    # === 配置 ===
    feat_path = "/mnt/cxh10/database/lizr/emotion/emotion2vec/iemocap_downstream_main/feats/train"  # 替换为你的特征路径
    model_path = "/mnt/cxh10/database/lizr/emotion/emotion2vec/iemocap_downstream_main/outputs/2025-10-23/14-29-17/model_2.pth"  # 第2折模型（fold=1 → model_2.pth）
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    emotion_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    emotion_labels = ['ang', 'hap', 'neu', 'sad']
    
    n_samples = [1085, 1023, 1151, 1031, 1241]  # Session1~5 的样本数
    fold = 1  # 第2折（0-indexed）
    
    test_idx_start = sum(n_samples[:fold])
    test_idx_end = test_idx_start + n_samples[fold]
    
    # === 加载数据 ===
    dataset = load_ssl_features(feat_path, emotion_dict)
    _, _, test_loader = train_valid_test_iemocap_dataloader(
        dataset,
        batch_size=128,
        test_start=test_idx_start,
        test_end=test_idx_end,
        eval_is_test=False
    )
    
    # === 加载模型 ===
    model = BaseModel(input_dim=768, num_classes=len(emotion_dict))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["net_input"]["feats"].to(device)
            padding_mask = batch["net_input"]["padding_mask"].to(device)
            labels = batch["labels"].to(device)
            
            emotion_out, _, _, _, _, _ = model(feats, padding_mask)
            preds = torch.argmax(emotion_out, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    # === 计算混淆矩阵（原始计数）===
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])

    # === 归一化：按真实标签（行）归一化 → 每行和为1 ===
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # === 处理除零情况（如果某类没有样本，避免 NaN）===
    cm_norm = np.nan_to_num(cm_norm)  # 将 NaN 替换为 0

    # === 绘图：使用归一化值，但保留原始整数作为辅助参考（可选）===
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.3f',           # 显示三位小数
        cmap='Blues',
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        cbar=False,
        annot_kws={"size": 28}
    )
        # 设置 x 轴和 y 轴标签字体大小
    plt.xticks(fontsize=28)      # 预测标签（下方）
    plt.yticks(fontsize=28, rotation=0)  # 真实标签（左侧），rotation=0 避免旋转
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Normalized Confusion Matrix - Fold 2 (IEMOCAP)')
    plt.tight_layout()

    # 保存图像
    output_file = "confusion_matrix_fold2_normalized.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Normalized confusion matrix saved to {output_file}")

    # 打印 UA（Unweighted Accuracy = 平均召回率 = 对角线均值）
    ua = cm_norm.diagonal().mean()
    print(f"Unweighted Accuracy (UA): {ua:.4f}")
    print("Normalized Confusion Matrix:")
    print(cm_norm)

if __name__ == "__main__":
    main()