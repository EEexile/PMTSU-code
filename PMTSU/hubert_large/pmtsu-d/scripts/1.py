#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile
from dataclasses import dataclass

from npy_append_array import NpyAppendArray
import soundfile as sf

import fairseq


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract HuBERT features for downstream tasks (e.g., IEMOCAP)"
    )
    # fmt: off
    parser.add_argument('--data', help='location of tsv files', required=True)
    parser.add_argument('--split', help='which split to read (e.g., train, test)', required=True)
    parser.add_argument('--checkpoint', type=str, help='path to HuBERT checkpoint (e.g., hubert_large_ll60k.pt)', required=True)
    parser.add_argument('--save-dir', help='where to save the output .npy and .lengths', required=True)
    parser.add_argument('--layer', type=int, default=23, 
                        help='which layer to extract (HuBERT large has 24 layers: 0-23). Default: 23 (last)')
    # fmt: on
    return parser


class HubertFeatureReader:
    def __init__(self, checkpoint, layer):
        # Load HuBERT model
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        self.model = models[0]
        self.model.eval()
        self.model.cuda()
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load audio and ensure it's 16kHz mono"""
        wav, sr = sf.read(fname)
        assert sr == 16000, f"Sample rate must be 16kHz, got {sr} in {fname}"
        if len(wav.shape) > 1:
            wav = wav[:, 0]  # take first channel if stereo
        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            # HuBERT expects input shape (B, T)
            source = source.view(1, -1)

            # Extract features from all layers
            logits = self.model.extract_features(
                source,
                padding_mask=None,
                mask=False,
                output_layer=self.layer + 1  # Fairseq uses 1-indexed layer output
            )
            # logits is a tuple: (features, extra)
            feats = logits[0]  # [1, T, C]
            return feats.squeeze(0).cpu()  # [T, C]


def get_iterator(args):
    tsv_path = osp.join(args.data, args.split + ".tsv")
    with open(tsv_path, "r") as fp:
        lines = fp.read().strip().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if line.strip()]

    num = len(files)
    reader = HubertFeatureReader(args.checkpoint, args.layer)

    def iterate():
        for fname in files:
            feats = reader.get_feats(fname)
            yield feats

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    save_path = osp.join(args.save_dir, args.split)
    npy_path = save_path + ".npy"
    if osp.exists(npy_path):
        os.remove(npy_path)
    npaa = NpyAppendArray(npy_path)

    generator, num = get_iterator(args)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f:
        for feats in tqdm.tqdm(iterator, total=num):
            print(len(feats), file=l_f)
            if len(feats) > 0:
                npaa.append(feats.numpy())


if __name__ == "__main__":
    main()