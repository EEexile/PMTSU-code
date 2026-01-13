#!/usr/bin/env python3 -u
import argparse
import os
import os.path as osp
import tqdm
import torch
from shutil import copyfile
from dataclasses import dataclass

from npy_append_array import NpyAppendArray
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def get_parser():
    parser = argparse.ArgumentParser(
        description="extract Whisper-large-v3 features for downstream tasks"
    )
    parser.add_argument('--data', help='location of tsv files', default='/mnt/cxh10/database/lizr/emotion/emotion2vec-main/iemocap_downstream/feats')
    parser.add_argument('--split', help='which split to read', default='train')
    parser.add_argument('--save-dir', help='where to save the output', default='/mnt/cxh10/database/lizr/emotion/emotion2vec-main/check/feats')
    parser.add_argument('--layer', type=int, default=-1,
                        help='which encoder layer to use (0-31). Default: -1 (last layer)')
    return parser


class WhisperFeatureReader:
    def __init__(self, layer=-1):
        model_path = "/mnt/cxh10/database/lizr/emotion/emotion2vec-main/whisper-large-v3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            output_hidden_states=True
        ).to(self.device).eval()
        self.layer = layer

    def read_audio(self, fname):
        """Load audio file, ensure 16kHz mono"""
        wav, sr = sf.read(fname)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # 转为单声道
        assert sr == 16000, f"Sample rate must be 16kHz, got {sr} in {fname}"
        return wav

    def get_feats(self, loc):
        wav = self.read_audio(loc)

        inputs = self.processor(
            wav, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)  # shape: (1, 80, T')

        with torch.no_grad():
            encoder_outputs = self.model.model.encoder(
                input_features=inputs,
                output_hidden_states=True
            )
            hidden_states = encoder_outputs.hidden_states  # tuple of (layer_0, ..., layer_32)

            selected_layer = hidden_states[self.layer]  # (1, seq_len, 1280)
            feats = selected_layer.squeeze(0).cpu()     # (seq_len, 1280)

        return feats


def get_iterator(args):
    tsv_path = osp.join(args.data, args.split) + ".tsv"
    with open(tsv_path, "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]

    num = len(files)
    reader = WhisperFeatureReader(layer=args.layer)

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
    # 清理旧文件
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