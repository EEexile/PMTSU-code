#把每条样本原本连续的 V/A/D 三维情感值（1-5 分）分别做 3-Means 聚类，转成 0/1/2 的离散三档，并写回同格式的标签文件。
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

def vad_3class(in_file: Path, out_file: Path):
    # 1. 读取原始文件
    lines = in_file.read_text(encoding='utf-8').splitlines()

    # 2. 收集所有 V A D
    vad_all = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        v, a, d = map(float, parts[2:5])   # 1-5 的连续值
        vad_all.append([v, a, d])
    vad_all = np.array(vad_all)

    # 3. 每一维单独 3-means
    labels_3d = np.empty_like(vad_all, dtype=int)
    for col in range(3):
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(vad_all[:, col:col+1])
        centers = km.cluster_centers_.ravel()
        order = np.argsort(centers)        # 0<1<2 
        mapping = {old: new for new, old in enumerate(order)}
        labels_3d[:, col] = [mapping[v] for v in km.labels_]

    # 4. 写回新文件
    with out_file.open('w', encoding='utf-8') as fw:
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            eid, emo = parts[0], parts[1]
            v_cls, a_cls, d_cls = labels_3d[idx]
            fw.write(f"{eid}\t{emo}\t{v_cls}\t{a_cls}\t{d_cls}\n")

    print(f"已生成 {out_file}")

# 用法
if __name__ == "__main__":
    root = Path("your_path")              # 改成你的目录
    vad_3class(root / "/mnt/cxh10/database/lizr/emotion/emotion2vec/iemocap_downstream_main/vad.lab",
              root / "/mnt/cxh10/database/lizr/emotion/emotion2vec/iemocap_downstream_main/kmeans.lab")