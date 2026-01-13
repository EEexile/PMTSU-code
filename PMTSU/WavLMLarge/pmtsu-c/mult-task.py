#基准代码 辅助任务为回归任务

import os
import sys
from pathlib import Path

import hydra 
from omegaconf import DictConfig

import torch
from torch import nn, optim

from data import load_ssl_features, train_valid_test_iemocap_dataloader

from utils import compute_unweighted_accuracy, compute_weighted_f1

import logging
import numpy as np
import torch.nn.functional as F
logger = logging.getLogger('IEMOCAP_Downstream')

import torch
from torch import nn
from torch.nn import MultiheadAttention


class BaseModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 768, 
                 hidden_dim: int = 256, 
                 num_classes: int = 4, 
                 num_heads: int = 4):
        super().__init__()
        
        # 1. 共享特征提取层
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. 任务适配层
        self.emo_adapter = nn.Linear(hidden_dim*2, hidden_dim)
        self.vad_adapters = nn.ModuleDict({
            'valence': nn.Linear(hidden_dim*2, hidden_dim),
            'arousal': nn.Linear(hidden_dim*2, hidden_dim),
            'dominance': nn.Linear(hidden_dim*2, hidden_dim)
        })
        
        # 3. 辅助任务输出层
        self.valence_head = nn.Linear(hidden_dim, 1)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.dominance_head = nn.Linear(hidden_dim, 1)
        
        # 4. 交叉注意力模块
        self.cross_attention = CrossTaskAttention(hidden_dim, num_heads)
        
        # 5. 门控融合模块
        self.fusion = GatedFusion(hidden_dim)
        
        # 6. 主任务输出层
        self.emotion_head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        # 共享特征提取
        shared_features = self.shared_layer(x)  # [B, L, 512]
        
        # 主任务特征
        emo_feat = self._get_emo_feature(shared_features, padding_mask)  # [B, 256]
        
        # 辅助任务处理
        vad_feats, valence_out, arousal_out, dominance_out = self._get_vad_features(
            shared_features, padding_mask)  # [B,3,256]
        
        # 交叉注意力
        attn_out, attn_weights = self.cross_attention(emo_feat, vad_feats)  # [B,3,256]
        
        # 门控融合
        fused_feature, fusion_weights = self.fusion(attn_out, emo_feat)  # [B,256]
        
        # 残差连接保证梯度流动
        fused_feature = fused_feature + emo_feat
        
        # 主任务预测
        emotion_out = self.emotion_head(fused_feature)
        
        return emotion_out, valence_out, arousal_out, dominance_out, attn_weights, fusion_weights
    
    def _get_emo_feature(self, features, mask):

        emo_feat = self.emo_adapter(features) 
        emo_feat = nn.functional.relu(emo_feat)
        """处理主任务特征"""
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            valid = emo_feat * (1 - mask_float)
            emo_feat = valid.sum(dim=1) / (1 - mask_float).sum(dim=1)
        else:
            emo_feat = emo_feat.mean(dim=1)
        return emo_feat 
    
    def _get_vad_features(self, features, mask):
        """处理三个VAD任务"""
        vad_feats = []
        outputs = []
        
        for task, adapter in self.vad_adapters.items():
            # 任务适配
            task_feat = adapter(features)  # [B, L, H]
            task_feat = nn.functional.relu(task_feat)
            
            # 处理掩码
            if mask is not None:
                mask_float = mask.unsqueeze(-1).float()
                valid_feat = task_feat * (1 - mask_float)
                avg_feat = valid_feat.sum(dim=1) / (1 - mask_float).sum(dim=1)
            else:
                avg_feat = task_feat.mean(dim=1)
            
            vad_feats.append(avg_feat)
            
            # 任务特定输出
            if task == 'valence':
                outputs.append(self.valence_head(avg_feat))
            elif task == 'arousal':
                outputs.append(self.arousal_head(avg_feat))
            else:
                outputs.append(self.dominance_head(avg_feat))
        
        return torch.stack(vad_feats, dim=1), *outputs

# ------------------ 模块实现 ------------------
class CrossTaskAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emo_query = nn.Linear(hidden_dim, hidden_dim)
        self.vad_proj = nn.Linear(hidden_dim, hidden_dim*2)  # Key+Value
        self.attn = MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        
    def forward(self, emo_feat, vad_feats):
        Q = self.emo_query(emo_feat).unsqueeze(1)  # [B,1,H]
        KV = self.vad_proj(vad_feats)  # [B,3,2H]
        K, V = torch.split(KV, self.hidden_dim, dim=-1)  # 使用 self.hidden_dim
        
        attn_out, attn_weights = self.attn(Q, K, V, need_weights=True)
        return attn_out, attn_weights 

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 门控网络：生成一个更新门
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # 输入是拼接后的特征
            nn.ReLU(),
            nn.Linear(64, hidden_dim),      # 输出与特征同维度
            nn.Sigmoid()                    # 输出0-1之间的门控值
        )
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, attn_out, emo_feat):
        """
        参数:
            attn_out: [B,1,H] -> 将被 squeeze 为 [B,H]
            emo_feat: [B,H] 主任务特征
        """
        attn_out = attn_out.squeeze(1)  # [B, H]
        
        # 1. 生成更新门
        # 将两个特征拼接起来作为门控网络的输入
        gate_input = torch.cat([emo_feat, attn_out], dim=-1)  # [B, 2*H]
        update_gate = self.update_gate(gate_input)  # [B, H]，每个特征维度一个门控值
        
        # 2. 门控融合：类似于 GRU 的更新机制
        # fused = (1 - update_gate) * emo_feat + update_gate * attn_out
        fused = emo_feat + update_gate * (attn_out - emo_feat)
        
        return self.transform(fused), update_gate  # 返回融合特征和门控值

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        print(f"{name}: {param}")
        total_params += param
    print(f"\nTotal number of parameters: {total_params}")

@torch.no_grad()
def validate_and_test(model, data_loader, device, num_classes):
    model.eval()

    # === 情感任务指标（保持不变）===
    emotion_correct, emotion_total = 0, 0
    emotion_unweighted_correct = [0] * num_classes
    emotion_unweighted_total = [0] * num_classes
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    # === VAD: 收集所有预测和真实值（用于全局 CCC）===
    all_valence_true, all_valence_pred = [], []
    all_arousal_true, all_arousal_pred = [], []
    all_dominance_true, all_dominance_pred = [], []

    # 注意力权重（可选）
    all_attn_weights = []
    all_fusion_weights = []

    for batch in data_loader:
        ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
        feats = net_input["feats"]
        speech_padding_mask = net_input["padding_mask"]
        vad_targets = batch["vad_labels"].to(device)  # [B, 3]

        feats = feats.to(device)
        speech_padding_mask = speech_padding_mask.to(device)
        labels = labels.to(device)

        emotion_out, valence_out, arousal_out, dominance_out, attn_weights, fusion_weights = model(feats, speech_padding_mask)
         
        # 情感预测（保持不变）
        _, emotion_pred = torch.max(emotion_out.data, 1)
        emotion_total += labels.size(0)
        emotion_correct += (emotion_pred == labels).sum().item()
        
        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = emotion_pred[i].item()
            emotion_unweighted_total[true_label] += 1
            if pred_label == true_label:
                emotion_unweighted_correct[true_label] += 1
                tp[true_label] += 1
            else:
                fp[pred_label] += 1
                fn[true_label] += 1

        # === VAD: 收集连续预测值和真实值 ===
        valence_pred = valence_out.squeeze(-1)  # [B]
        arousal_pred = arousal_out.squeeze(-1)
        dominance_pred = dominance_out.squeeze(-1)

        all_valence_true.append(vad_targets[:, 0])
        all_valence_pred.append(valence_pred)
        all_arousal_true.append(vad_targets[:, 1])
        all_arousal_pred.append(arousal_pred)
        all_dominance_true.append(vad_targets[:, 2])
        all_dominance_pred.append(dominance_pred)

        # 收集权重（可选）
        all_attn_weights.append(attn_weights.squeeze(1).cpu().numpy())
        all_fusion_weights.append(fusion_weights.cpu().numpy())

    # === 情感指标（不变）===
    emotion_wa = emotion_correct / emotion_total * 100
    emotion_ua = compute_unweighted_accuracy(emotion_unweighted_correct, emotion_unweighted_total) * 100
    weighted_f1 = compute_weighted_f1(tp, fp, fn, emotion_unweighted_total) * 100

    # === VAD: 计算 CCC ===
    def compute_ccc(true_list, pred_list):
        if len(true_list) == 0:
            return 0.0
        true_all = torch.cat(true_list, dim=0)
        pred_all = torch.cat(pred_list, dim=0)
        return concordance_correlation_coefficient(true_all, pred_all).item()

    valence_ccc = compute_ccc(all_valence_true, all_valence_pred)
    arousal_ccc = compute_ccc(all_arousal_true, all_arousal_pred)
    dominance_ccc = compute_ccc(all_dominance_true, all_dominance_pred)
    vad_ccc = [valence_ccc, arousal_ccc, dominance_ccc]

    # （可选）打印注意力分析
    if all_attn_weights:
        all_attn_weights = np.concatenate(all_attn_weights, axis=0)
        avg_attn_weights = all_attn_weights.mean(axis=0)
        print(f"\n📊 Cross-Attention Weights: Valence={avg_attn_weights[0]:.4f}, "
              f"Arousal={avg_attn_weights[1]:.4f}, Dominance={avg_attn_weights[2]:.4f}")

    return emotion_wa, emotion_ua, weighted_f1, vad_ccc

def concordance_correlation_coefficient(y_true, y_pred):
    """
    Compute CCC for two 1D tensors.
    Returns a scalar CCC value.
    """
    if y_true.numel() < 2:
        return torch.tensor(0.0, device=y_true.device)
    
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    
    var_true = torch.var(y_true, unbiased=False)
    var_pred = torch.var(y_pred, unbiased=False)
    
    cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc

def train_one_epoch(model, optimizer, criterion, train_loader, device, task_weights):
    model.train()
    total_loss = 0
    emotion_loss_total = 0
    # 分别记录三个VAD任务的损失
    valence_loss_total = 0
    arousal_loss_total = 0
    dominance_loss_total = 0
    
    for batch in train_loader:
        ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
        feats = net_input["feats"]
        speech_padding_mask = net_input["padding_mask"]

        feats = feats.to(device)
        speech_padding_mask = speech_padding_mask.to(device)
        labels = labels.to(device)
        vad_labels = batch["vad_labels"].to(device)  # [B, 3]
        
        optimizer.zero_grad()
        
        # 模型前向传播
        emotion_out, valence_out, arousal_out, dominance_out,_, _ = model(feats, speech_padding_mask)
        
        # 计算情感任务损失
        emotion_loss = criterion(emotion_out, labels.long())
        
        valence_pred = valence_out.squeeze(-1)  # [B, 1] -> [B]
        valence_loss = F.mse_loss(valence_pred, vad_labels[:, 0])        

        arousal_pred = arousal_out.squeeze(-1)
        arousal_loss = F.mse_loss(arousal_pred, vad_labels[:, 1])

        dominance_pred = dominance_out.squeeze(-1)
        dominance_loss = F.mse_loss(dominance_pred, vad_labels[:, 2])

        # 加权总损失
        total_loss_batch = (
            task_weights[0] * emotion_loss +
            task_weights[1] * valence_loss +
            task_weights[2] * arousal_loss +
            task_weights[3] * dominance_loss
        )
        
        total_loss_batch.backward()
        optimizer.step()
        
        # 累加各项损失
        total_loss += total_loss_batch.item()
        emotion_loss_total += emotion_loss.item()
        valence_loss_total += valence_loss.item()
        arousal_loss_total += arousal_loss.item()
        dominance_loss_total += dominance_loss.item()
    
    # 计算平均损失
    num_batches = len(train_loader)
    return (
        total_loss / num_batches, 
        emotion_loss_total / num_batches,
        [  # 返回VAD三个维度的单独损失
            valence_loss_total / num_batches,
            arousal_loss_total / num_batches,
            dominance_loss_total / num_batches
        ]
    )

@hydra.main(config_path='config', config_name='default.yaml')
def train_iemocap(cfg: DictConfig):
    # 情感标签映射
    emotion_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}

    
    n_samples = [1085, 1023, 1151, 1031, 1241]  # Session1, 2, 3, 4, 5
    idx_sessions = [0, 1, 2, 3, 4]

    test_wa_avg, test_ua_avg, test_vad_ccc_avg = 0., 0., [0., 0., 0.]
    test_f1_avg = 0.0

    for fold in idx_sessions:  # extract the $fold$th as test set
        # torch.manual_seed(cfg.common.seed)    
        #随机种子设立
        seed = cfg.common.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
        logger.info(f"------Now it's {fold+1}th fold------")
        
        val_wa_history = [] 
        prev_emo_loss = 1.0  # 初始主任务损失
        prev_vad_losses = [1.0, 1.0, 1.0]  # 初始VAD损失  
        
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        
        # 加载数据集
        dataset = load_ssl_features(cfg.dataset.feat_path, emotion_dict)

        test_len = n_samples[fold] 
        test_idx_start = sum(n_samples[:fold])
        test_idx_end = test_idx_start + test_len 
        train_loader, val_loader, test_loader = train_valid_test_iemocap_dataloader(
            dataset,
            cfg.dataset.batch_size,
            test_idx_start,
            test_idx_end,
            eval_is_test=cfg.dataset.eval_is_test,
        )

        model = BaseModel(input_dim=1024, num_classes=len(emotion_dict))
        model = model.to(device)

        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=cfg.optimization.lr, 
            weight_decay=cfg.optimization.get('weight_decay', 1e-4)  # 添加权重衰减
        )
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=cfg.optimization.lr, 
            max_lr=3e-3, 
            step_size_up=20,
            cycle_momentum=False 
        )
        criterion = nn.CrossEntropyLoss()

        # 定义训练阶段参数
        phase1_epochs = 10  # 阶段1的epoch数
        phase2_total_epochs = max(0, cfg.optimization.epoch - phase1_epochs)
        
        # 计算阶段2的解冻步骤
        if phase2_total_epochs > 0:
            step_interval = phase2_total_epochs // 3
            step1_epoch = phase1_epochs  # 开始解冻vad_adapters
            step2_epoch = phase1_epochs + step_interval  # 开始解冻cross_attention
            step3_epoch = phase1_epochs + 2 * step_interval  # 开始解冻fusion
        else:
            step1_epoch = step2_epoch = step3_epoch = phase1_epochs

        best_val_wa = 0
        best_val_wa_epoch = 0
        save_dir = os.path.join(str(Path.cwd()), f"model_{fold+1}.pth")
        
        for epoch in range(cfg.optimization.epoch):
            
            # 动态调整参数冻结状态和任务权重
            if epoch < phase1_epochs:
                # 阶段1: 仅训练主任务，冻结所有VAD相关参数
                # 冻结VAD适配器
                for param in model.vad_adapters.parameters():
                    param.requires_grad = False
                # 冻结VAD输出层
                for param in model.valence_head.parameters():
                    param.requires_grad = False
                for param in model.arousal_head.parameters():
                    param.requires_grad = False
                for param in model.dominance_head.parameters():
                    param.requires_grad = False
                # 冻结交叉注意力
                for param in model.cross_attention.parameters():
                    param.requires_grad = False
                # 冻结门控融合模块
                for param in model.fusion.parameters():
                    param.requires_grad = False
                
                # 仅使用情感任务损失
                task_weights = [1.0, 0.0, 0.0, 0.0]
                logger.info(f"Epoch {epoch+1}: PHASE 1 - Only training emotion task, freezing all VAD components")
            else:   
               # 阶段2: 渐进式解冻参数
                if epoch < step2_epoch:
                    # 步骤1: 解冻vad_adapters（任务适配层）
                    for param in model.vad_adapters.parameters():
                        param.requires_grad = True
                    for param in model.valence_head.parameters():
                        param.requires_grad = True
                    for param in model.arousal_head.parameters():
                        param.requires_grad = True
                    for param in model.dominance_head.parameters():
                        param.requires_grad = True
                    # 保持cross_attention和fusion冻结
                    for param in model.cross_attention.parameters():
                        param.requires_grad = False
                    for param in model.fusion.parameters():
                        param.requires_grad = False
                    logger.info(f"Epoch {epoch+1}: PHASE 2 - Step 1: Unfrozen VAD adapters")
                elif epoch < step3_epoch:
                    # 步骤2: 解冻cross_attention（注意力模块）
                    for param in model.cross_attention.parameters():
                        param.requires_grad = True
                    # 保持fusion冻结
                    for param in model.fusion.parameters():
                        param.requires_grad = False
                    logger.info(f"Epoch {epoch+1}: PHASE 2 - Step 2: Unfrozen cross-attention")
                else:
                    # 步骤3: 解冻fusion（门控模块）
                    for param in model.fusion.parameters():
                        param.requires_grad = True
                    logger.info(f"Epoch {epoch+1}: PHASE 2 - Step 3: Unfrozen fusion module")
                
     
                phase2_epoch = epoch - phase1_epochs
                if phase2_total_epochs > 0:
                    ratio = min(phase2_epoch / phase2_total_epochs, 1.0)
                else:
                    ratio = 1.0
                
                w_emotion = 1 - 0.4 * ratio  
                total_vad_weight = 0.0 + 0.4 * ratio  
                w_v = w_a = w_d = total_vad_weight / 3.0  
                
                task_weights = [w_emotion, w_v, w_a, w_d]
                logger.info(f"Epoch {epoch+1}: PHASE 2 - Task Weights: Emo={w_emotion:.2f}, VAD Total={total_vad_weight:.2f} (V={w_v:.2f}, A={w_a:.2f}, D={w_d:.2f})")
            
            # 训练时传入VAD权重
            total_loss, emotion_loss, vad_loss = train_one_epoch(
                model, optimizer, criterion, train_loader, device, 
                task_weights
            )
            scheduler.step()
            v_loss, a_loss, d_loss = vad_loss 
            
            # 更新损失记录
            prev_emo_loss = emotion_loss
            prev_vad_losses = [v_loss, a_loss, d_loss] 

            # 验证
            val_wa, val_ua, val_f1, val_vad_ccc = validate_and_test(
                model, val_loader, device, num_classes=len(emotion_dict)
            )
            val_wa_history.append(val_wa)

            # 日志输出权重信息
            logger.info(f"Epoch {epoch+1} Task Weights: "
                        f"Emo={task_weights[0]:.2f}, "
                        f"Val={task_weights[1]:.2f}, "
                        f"Aro={task_weights[2]:.2f}, "
                        f"Dom={task_weights[3]:.2f}")

            
            if val_wa > best_val_wa:
                best_val_wa = val_wa
                best_val_wa_epoch = epoch
                torch.save(model.state_dict(), save_dir)

            # 修改后的日志语句
            logger.info(f"Epoch {epoch+1}: "
                        f"Total Loss: {total_loss:.4f}, "
                        f"Emotion Loss: {emotion_loss:.4f}, "
                        f"VAD Loss: V={vad_loss[0]:.4f}, A={vad_loss[1]:.4f}, D={vad_loss[2]:.4f}, "  # 分别访问列表元素
                        f"Val WA: {val_wa:.3f}%, "
                        f"Val UA: {val_ua:.3f}%, "
                        f"Val F1: {val_f1:.3f}%, "  # ✅ 新增 F1
                        f"Val VAD CCC: V={val_vad_ccc[0]:.4f}, A={val_vad_ccc[1]:.4f}, D={val_vad_ccc[2]:.4f}")

        # 测试
        ckpt = torch.load(save_dir)
        model.load_state_dict(ckpt, strict=True)
        test_wa, test_ua,test_f1, test_vad_ccc = validate_and_test(
            model, test_loader, device, num_classes=len(emotion_dict)
        )
        
        logger.info(f"The {fold+1}th Fold at epoch {best_val_wa_epoch + 1}: "
                    f"Test WA {test_wa:.3f}%, "
                    f"Test UA {test_ua:.3f}%, "
                    f"Test F1 {test_f1:.3f}%, "  # ✅ 新增 F1
                    f"Test VAD CCC: V={test_vad_ccc[0]:.4f}, A={test_vad_ccc[1]:.4f}, D={test_vad_ccc[2]:.4f}")
        
        test_wa_avg += test_wa
        test_ua_avg += test_ua
        test_f1_avg += test_f1
        test_vad_ccc_avg = [test_vad_ccc_avg[i] + test_vad_ccc[i] for i in range(3)]

    # 计算平均指标
    num_folds = len(idx_sessions)
    test_wa_avg /= num_folds
    test_ua_avg /= num_folds
    test_vad_ccc_avg = [ccc / num_folds for ccc in test_vad_ccc_avg]    


    logger.info(f"Average Results: "
                f"WA: {test_wa_avg:.3f}%, "
                f"UA: {test_ua_avg:.3f}%, "
                f"F1: {test_f1_avg / len(idx_sessions):.3f}%, " 
                f"VAD CCC: V={test_vad_ccc_avg[0]:.4f}, A={test_vad_ccc_avg[1]:.4f}, D={test_vad_ccc_avg[2]:.4f}")

if __name__ == '__main__':
    train_iemocap()

