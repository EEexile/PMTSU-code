#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import librosa
import torch
from tqdm import tqdm
from npy_append_array import NpyAppendArray
from transformers import AutoModel  # 只需要 AutoModel

def load_tsv_manifest(tsv_path):
    with open(tsv_path, 'r') as f:
        lines = f.read().strip().split('\n')
    root = lines[0].strip()
    audio_paths = []
    for line in lines[1:]:
        if not line.strip():
            continue
        rel_path = line.split('\t')[0]
        audio_paths.append(osp.join(root, rel_path))
    return audio_paths

def main():
    manifest_path = "/mnt/cxh10/database/lizr/emotion/emotion2vec-main/iemocap_downstream/feats/train.tsv"
    save_dir = "/mnt/cxh10/database/lizr/emotion/emotion2vec-main/iemocap_downstream/feats"
    split = "train"
    os.makedirs(save_dir, exist_ok=True)

    audio_paths = load_tsv_manifest(manifest_path)
    print(f"Loaded {len(audio_paths)} audio files from {manifest_path}")

    # === 加载 WavLM-large（仅模型，不用 processor）===
    model_id = "/mnt/cxh10/database/lizr/emotion/emotion2vec-main/wavlm-large"
    print(f"Loading WavLM-large from: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        # use_safetensors=True  # 如果你有 .safetensors
    ).to(device)
    model.eval()
    print("WavLM-large loaded.")

    # === 准备输出文件 ===
    save_prefix = osp.join(save_dir, split)
    npy_path = save_prefix + ".npy"
    lengths_path = save_prefix + ".lengths"

    if osp.exists(npy_path):
        os.remove(npy_path)
    npaa = NpyAppendArray(npy_path)

    # === 特征提取 ===
    with open(lengths_path, 'w') as lf:
        for audio_path in tqdm(audio_paths, desc="Extracting WavLM features"):
            if not osp.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # 1. 加载 16kHz 单声道音频
            audio, sr = librosa.load(audio_path, sr=16000)

            # 2. 转为 tensor: shape (1, T)
            input_values = torch.tensor(audio).unsqueeze(0).to(device, dtype=torch_dtype)

            # 3. 前向传播（无需 attention mask）
            with torch.no_grad():
                outputs = model(input_values)
                feats = outputs.last_hidden_state  # (1, T', 1024)

            # 4. 转为 numpy
            feats = feats.squeeze(0).cpu().float().numpy()  # (T', 1024)

            # 5. 保存
            print(len(feats), file=lf)
            if len(feats) > 0:
                npaa.append(feats)

    print(f"WavLM features saved to {npy_path}")
    print(f"Lengths saved to {lengths_path}")

if __name__ == "__main__":
    main()