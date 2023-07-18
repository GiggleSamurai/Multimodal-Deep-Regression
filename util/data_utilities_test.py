import os
import json
import torch
import itertools
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


# from data.dataset import Dataset
# from torch.utils.data import DataLoader
# from data.tensor_helpers import ints_to_tensor, pad_tensors


def process_data(input_type, addition_parameters=None, verbose=False, device='cpu', skip_frames=False, frames_to_skip=5, shrink=1, normalize=False):
    """
    For this implementation to work you'll need to have the videos loaded into a directory under
    '../data/video_packs/input_type'
    """
    
    assert input_type in get_valid_input_types(), 'Current implementation only set to process initial_1000, 1k and 5k video packs'

    sys.path.append("..")
    top_level_path = f'../data/video_packs/{input_type}'
    video_list = os.listdir(top_level_path)
    
    if addition_parameters is None:
        first_n_videos = len(video_list)
    else:
        if 'first_n_videos' in addition_parameters.keys():
            first_n_videos = addition_parameters['first_n_videos']
        else:
            first_n_videos = len(video_list)
    
    if not verbose:
        progress = tqdm(total=first_n_videos)

    video_views = get_video_play_count(input_type=input_type)
    
    # Removing this
    # if verbose:
    #     print(video_views)

    x_dir = f"../data/x_tensors/{input_type}/"
    y_dir = f"../data/y_tensors/{input_type}/"

    os.makedirs(x_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)

    processed_videos, video_count = 0, 0 
    while processed_videos < first_n_videos or video_count == len(video_list) - 1:
        tiktok_video = video_list[video_count]
        video_path = f"{top_level_path}/{tiktok_video}"
        tiktok_video_id = tiktok_video.split(sep='.')[0]

        # Handling case when video doesn't have view count
        if tiktok_video_id not in video_views.keys():
            if verbose:
                print(f'No video views for {tiktok_video_id}, moving to next video.')
            video_count += 1
            continue

        if verbose:
            print(f'Currently processing: {tiktok_video}')

        
        vf, af, info, meta = process_video(video_object_path=video_path, device=device)
        # Reshaping tensor
        frames, n_channels, height, width = vf.shape
        vf = torch.reshape(vf, (n_channels, frames, height, width))

        if skip_frames:
            if verbose:
                print(f'Downsampling video tensors for every {frames_to_skip} frame')
                print(f'Original tensor size: {(n_channels, frames, height, width)}')
            frame_idx = [True if i % frames_to_skip == 0 else False for i in range(frames)]
            vf = vf[:, frame_idx, : ,:]

            if verbose:
                print(f'New tensor size: {vf.shape}')
        
        # Need to normalize the tensor
        #vf = nn.functional.normalize(vf, dim=(0, 1))
        if normalize:
            vf = vf/ 255.0

        # resize the tensor to 1024x576
        vf = resize_tensor(vf)
        if shrink > 1:
            vf = shrink_video(vf,shrink=shrink)
        if verbose:
                print(f'Resize to tensor size: {vf.shape}')
                
        x_file_path = f"{x_dir}{tiktok_video_id}_x_tensor.pt"
        y_file_path = f"{y_dir}{tiktok_video_id}_y_tensor.pt"

        # Save the x & y tensors
        with open(x_file_path, 'wb') as x_file, open(y_file_path, 'wb') as y_file:
            # Converting target video play count to tensor and adding dimension 
            if verbose:
                print(tiktok_video_id)
                print(video_views[tiktok_video_id])

            y_data = torch.as_tensor([video_views[tiktok_video_id]]).float().unsqueeze(1)
            
            torch.save(vf, x_file)
            torch.save(y_data, y_file)

        if verbose:
            print(f'Done processing: {video_path}.')
            print(f'X Tensor ({vf.shape}) saved under: {x_file_path}.')
            print(f'Y Tensor ({y_data.shape}) saved under: {y_file_path}.')
            print(f'metadata:\n{meta}\n\n')

        processed_videos += 1
        video_count += 1

        if not verbose:
            progress.update(1) 
    if not verbose:
        progress.close()

def get_video_play_count(input_type):
    """
    Retrieving video views
    """
    if input_type == 'initial_1000':
        f = open('../data/trending.json', encoding="utf8")
        data = json.load(f)
        video_metadata_list = data['collector']

        video_views = {}
        for video_metadata in video_metadata_list:
            video_views[video_metadata['id']] = video_metadata['playCount']
    elif input_type in ['video_pack_1000', 'video_pack_5000']:
        f = open('../data/tiktok_meta_data.json', encoding="utf8")
        video_metadata_list = json.load(f)
        video_views = {}
        for video_metadata in video_metadata_list:
            video_views[video_metadata['id']] = video_metadata['views_k']
    else:
        raise Exception(f'Retrieval of video play count not implemented for {input_type} dataset.')

    return video_views


def process_video(video_object_path, start=0, end=None, read_video=True, read_audio=False, device='cpu'):
    """
    Based off of https://pytorch.org/vision/main/auto_examples/plot_video_api.html#sphx-glr-auto-examples-plot-video-api-py
    """
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )

    video_object = torchvision.io.VideoReader(video_object_path, "video")

    video_frames = torch.empty(0) # .to(device)
    video_pts = []
    if read_video:
        video_object.set_current_stream("video")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
            frames.append(frame['data'])
            video_pts.append(frame['pts'])
        if len(frames) > 0:
            video_frames = torch.stack(frames, 0)

    audio_frames = torch.empty(0) # .to(device)
    audio_pts = []
    if read_audio:
        video_object.set_current_stream("audio")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
            frames.append(frame['data'])
            audio_pts.append(frame['pts'])
        if len(frames) > 0:
            audio_frames = torch.cat(frames, 0)

    return video_frames, audio_frames, (video_pts, audio_pts), video_object.get_metadata()


def get_valid_input_types():
    """
    initial_1000 dataset: https://www.kaggle.com/datasets/erikvdven/tiktok-trending-december-2020?resource=download
    """
    return ['initial_1000', 'video_pack_1000', 'video_pack_5000']


def get_base_tensor_directories(input_type):
    return f"../data/x_tensors/{input_type}/", f"../data/y_tensors/{input_type}/"


def generate_batch(batch):
    
    # max depth of each batch
    max_d = max([x.shape[1] for x, y in batch])
    padded_x = []
    y_batch = []

    for x, y in batch:
        d = x.shape[1]
        
        # ConstantPad3d (left, right, top, bottom, front, back)
        padding = nn.ConstantPad3d((0, 0, 0, 0, 0, max_d - d), 0)
        padded_x.append(padding(x))
        y_batch.append(y)

    x = torch.stack(padded_x)
    y = torch.tensor(y_batch)
    return x, y

def resize_tensor(input_tensor):
    original_height = input_tensor.shape[2]
    original_width = input_tensor.shape[3]

    if original_height == 1024 and original_width == 576:
        return input_tensor

    # resize tensor and keep to input ratio
    new_width = 576
    aspect_ratio = original_height / original_width

    # new height can not be bigger than 1024
    new_height = min(1024,int(new_width * aspect_ratio))
    resized_tensor = nn.functional.interpolate(input_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # fill rest with 0 padding
    padding_height = 1024 - resized_tensor.shape[2]
    padding_width = 576 - resized_tensor.shape[3]

    # padding (left, right, top, bottom)
    padding = (padding_width//2, padding_width//2, padding_height//2, padding_height//2)
    padded_tensor = nn.functional.pad(resized_tensor, padding, "constant", 0)
    return padded_tensor

def shrink_video(input_tensor,shrink=1):
    new_height = 1024//shrink
    new_width = 576//shrink
    resized_tensor = nn.functional.interpolate(input_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return resized_tensor