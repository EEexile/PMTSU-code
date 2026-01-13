#生成2 3 4 means聚类 并且比较轮廓系数
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

def vad_multik_means(in_file: Path, out_dir: Path, ks=(3, 4, 5)):
    """
    对 VAD 每个维度分别进行 K=2,3,4 的聚类，保存离散标签，并输出轮廓系数。
    
    Args:
        in_file: 原始 VAD 标签文件路径
        out_dir: 输出目录（会生成 vad_k2.lab, vad_k3.lab, vad_k4.lab）
        ks: 要尝试的聚类数，默认 (2, 3, 4)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 读取原始文件
    lines = in_file.read_text(encoding='utf-8').splitlines()
    
    # 2. 收集所有 V A D
    vad_all = []
    valid_indices = []  # 记录非空行的索引，用于对齐
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        try:
            v, a, d = map(float, parts[2:5])
            vad_all.append([v, a, d])
            valid_indices.append(idx)
        except ValueError:
            continue
    vad_all = np.array(vad_all)  # shape: (N, 3)

    # 维度名称
    dim_names = ['Valence', 'Arousal', 'Dominance']

    # 存储轮廓系数结果
    results = {}

    # 3. 对每个 K 进行处理
    for k in ks:
        print(f"\n=== 处理 K={k} ===")
        labels_3d = np.empty_like(vad_all, dtype=int)
        sil_scores = {}

        for col in range(3):
            X = vad_all[:, col:col+1]  # shape (N, 1)
            
            # 聚类
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            labels = km.labels_
            
            # 计算轮廓系数（注意：k=1 时无法计算，但这里 k>=2）
            sil = silhouette_score(X, labels)
            sil_scores[dim_names[col]] = sil
            print(f"{dim_names[col]}: 轮廓系数 = {sil:.4f}")

            # 按聚类中心排序，映射为 0,1,...,k-1（保持语义顺序）
            centers = km.cluster_centers_.ravel()
            order = np.argsort(centers)  # 从小到大
            mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
            labels_ordered = np.array([mapping[l] for l in labels])
            labels_3d[:, col] = labels_ordered

        results[k] = sil_scores

        # 4. 写回文件（只覆盖有效行，空行跳过）
        out_file = out_dir / f"vad_k{k}.lab"
        with out_file.open('w', encoding='utf-8') as fw:
            line_idx = 0
            for orig_idx, line in enumerate(lines):
                if not line.strip():
                    continue
                if line_idx >= len(valid_indices):
                    break
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                eid, emo = parts[0], parts[1]
                v_cls, a_cls, d_cls = labels_3d[line_idx]
                fw.write(f"{eid}\t{emo}\t{v_cls}\t{a_cls}\t{d_cls}\n")
                line_idx += 1

        print(f"已生成 {out_file}")

    # 5. 打印汇总结果
    print("\n=== 轮廓系数汇总 ===")
    print("K\tValence\tArousal\tDominance")
    for k in ks:
        s = results[k]
        print(f"{k}\t{s['Valence']:.4f}\t{s['Arousal']:.4f}\t{s['Dominance']:.4f}")

    # 可选：保存轮廓系数到文件
    sil_file = out_dir / "silhouette_scores.txt"
    with sil_file.open('w') as f:
        f.write("K\tValence\tArousal\tDominance\n")
        for k in ks:
            s = results[k]
            f.write(f"{k}\t{s['Valence']:.4f}\t{s['Arousal']:.4f}\t{s['Dominance']:.4f}\n")
    print(f"轮廓系数已保存至 {sil_file}")

# 用法
if __name__ == "__main__":
    in_file = Path("/mnt/cxh10/database/lizr/emotion/emotion2vec/iemocap_downstream_main/vad.lab")
    out_dir = Path("/mnt/cxh10/database/lizr/emotion/emotion2vec/iemocap_downstream_main/kmeans_vad")
    vad_multik_means(in_file, out_dir, ks=(2, 3, 4, 5))