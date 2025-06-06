import pdb
import torch as th
import math
import numpy as np
import torch
from run_on_video.video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from run_on_video.preprocessing import Preprocessing
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
from run_on_video import clip
import argparse

#################################
@torch.no_grad()
def vid2clip(model, vid_path, output_file, 
             model_version="ViT-B/32", output_feat_size=512,
             clip_len=2, overwrite=True, num_decoding_thread=4, half_precision=False):
    dataset = VideoLoader(
        vid_path,
        framerate=1/clip_len,
        size=224,
        centercrop=True,
        overwrite=overwrite,
        model_version=model_version
    )
    n_dataset = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_decoding_thread,
        sampler=None,
    )
    preprocess = Preprocessing()
    device_id = next(model.parameters()).device

    totatl_num_frames = 0
    with th.no_grad():
        for k, data in enumerate(tqdm(loader)):
            input_file = data['input'][0]
            if os.path.isfile(output_file):
                # print(f'Video {input_file} already processed.')
                continue
            elif not os.path.isfile(input_file):
                print(f'{input_file}, does not exist.\n')
            elif len(data['video'].shape) > 4:
                video = data['video'].squeeze(0)
                if len(video.shape) == 4:
                    video = preprocess(video)
                    n_chunk = len(video)
                    vid_features = th.zeros((n_chunk, output_feat_size), device=device_id)
                    n_iter = int(math.ceil(n_chunk))
                    for i in range(n_iter):
                        min_ind = i
                        max_ind = (i + 1)
                        video_batch = video[min_ind:max_ind].to(device_id)
                        batch_features = model.encode_image(video_batch)
                        vid_features[min_ind:max_ind] = batch_features
                    vid_features = vid_features.cpu().numpy()
                    if half_precision:
                        vid_features = vid_features.astype('float16')
                    totatl_num_frames += vid_features.shape[0]
                    # safeguard output path before saving
                    dirname = os.path.dirname(output_file)
                    if not os.path.exists(dirname):
                        print(f"Output directory {dirname} does not exists, creating...")
                        os.makedirs(dirname)
                    np.savez(os.path.join(output_file, 'vid.npz'), features=vid_features)
            else:
                print(f'{input_file}, failed at ffprobe.\n')
    print(f"Total number of frames: {totatl_num_frames}")
    return vid_features

def txt2clip(model, text, output_file):
    device_id = next(model.parameters()).device
    encoded_texts = clip.tokenize(text).to(device_id)
    text_feature = model.encode_text(encoded_texts)['last_hidden_state']
    valid_lengths = (encoded_texts != 0).sum(1).tolist()[0]
    text_feature = text_feature[0, :valid_lengths].detach().cpu().numpy()
    
    np.savez(os.path.join(output_file, 'txt.npz'), features=text_feature)
    return text_feature
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--vid_path', type=str, default='/data/home/qinghonglin/dataset/charades/videos/Charades_v1_480/0A8CF.mp4')
  parser.add_argument('--text', nargs='+', type=str, default='a boy is drinking.')
  parser.add_argument('--save_dir', type=str, default='./tmp')
  args = parser.parse_args()
