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
import fairseq
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="extract HuBERT features for downstream tasks"
    )
    # fmt: off
    parser.add_argument('--data', help='location of tsv files', default='/mnt/cxh10/database/lizr/emotion/hubert_large/iemocap_downstream/feats')
    parser.add_argument('--split', help='which split to read', default='train')
    parser.add_argument('--checkpoint', type=str, help='path to HuBERT checkpoint', default='/mnt/cxh10/database/lizr/emotion/hubert_large/model/hubert_large_ll60k.pt')
    parser.add_argument('--save-dir', help='where to save the output', default='/mnt/cxh10/database/lizr/emotion/hubert_large/iemocap_downstream/feats')
    parser.add_argument('--layer', type=int, default=23, 
                        help='which layer to use. Large: 0-23 (24 layers). Default: 23 (last)')
    # fmt: on
    return parser


class HubertFeatureReader:
    def __init__(self, checkpoint, layer):
        # Load model
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        model = models[0]
        model.eval()
        model.cuda()

        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        channel = sf.info(fname).channels
        assert sr == 16000, f"Sample rate should be 16kHz, but got {sr} in file {fname}"
        assert channel == 1, f"Channel should be 1, but got {channel} in file {fname}"
        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            # Extract features from specified layer
            features, _ = self.model.extract_features(
                source=source,
                padding_mask=None,
                mask=False,
                output_layer=self.layer + 1  # fairseq uses 1-indexed layers
            )
            # features shape: [1, T, D]
            return features.squeeze(0).cpu()  # [T, D]


def get_iterator(args):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]

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
    if osp.exists(save_path + ".npy"):
        os.remove(save_path + ".npy")
    npaa = NpyAppendArray(save_path + ".npy")

    generator, num = get_iterator(args)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f:
        for feats in tqdm.tqdm(iterator, total=num):
            print(len(feats), file=l_f)
            if len(feats) > 0:
                npaa.append(feats.numpy())


if __name__ == "__main__":
    main()