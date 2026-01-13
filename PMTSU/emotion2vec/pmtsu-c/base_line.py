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

class VanillaMTL(nn.Module):
    def __init__(self, input_dim=768, shared_dim=512, hidden_head=256, num_classes=4):
        super().__init__()
        # 共享层: input_dim -> shared_dim (e.g., 768 -> 512)
        self.shared_proj = nn.Linear(input_dim, shared_dim)
        
        # Emotion head: shared_dim -> hidden_head -> num_classes
        self.emotion_head = nn.Sequential(
            nn.Linear(shared_dim, hidden_head),
            nn.ReLU(),
            nn.Linear(hidden_head, num_classes)
        )
        
        # VAD heads: each is shared_dim -> hidden_head -> 1
        self.valence_head = nn.Sequential(
            nn.Linear(shared_dim, hidden_head),
            nn.ReLU(),
            nn.Linear(hidden_head, 1)
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(shared_dim, hidden_head),
            nn.ReLU(),
            nn.Linear(hidden_head, 1)
        )
        self.dominance_head = nn.Sequential(
            nn.Linear(shared_dim, hidden_head),
            nn.ReLU(),
            nn.Linear(hidden_head, 1)
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        # Pooling: [B, L, D] -> [B, D]
        if padding_mask is not None:
            mask_float = padding_mask.unsqueeze(-1).float()
            valid = x * (1 - mask_float)
            x_pooled = valid.sum(dim=1) / (1 - mask_float).sum(dim=1)
        else:
            x_pooled = x.mean(dim=1)
        
        # Shared representation
        shared = self.shared_proj(x_pooled)  # [B, 512]
        shared = nn.functional.relu(shared)
        
        # Task-specific heads
        emotion_out = self.emotion_head(shared)
        valence_out = self.valence_head(shared)
        arousal_out = self.arousal_head(shared)
        dominance_out = self.dominance_head(shared)
        
        # Return compatible with your validate_and_test
        return emotion_out, valence_out, arousal_out, dominance_out, None, None

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

        # model = BaseModel(input_dim=768, num_classes=len(emotion_dict))
        model = VanillaMTL(input_dim=768, shared_dim=512, hidden_head=256, num_classes=len(emotion_dict))
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



        best_val_wa = 0
        best_val_wa_epoch = 0
        save_dir = os.path.join(str(Path.cwd()), f"model_{fold+1}.pth")
        
        for epoch in range(cfg.optimization.epoch):
            
            task_weights = [1, 1, 1, 1]      
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
                    f"Test CCC: V={test_vad_ccc[0]:.4f}, A={test_vad_ccc[1]:.4f}, D={test_vad_ccc[2]:.4f}")
        
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

