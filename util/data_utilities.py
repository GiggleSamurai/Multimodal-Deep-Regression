import os
import json
import torch
import itertools
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import sys


# from data.dataset import Dataset
# from torch.utils.data import DataLoader
# from data.tensor_helpers import ints_to_tensor, pad_tensors


def process_data(input_type, addition_parameters=None, verbose=False, device='CPU'):
    
    assert input_type in ['initial_1000'], 'Current implementation only set to process 1000 video dataset (initial_1000)'

    if input_type == 'initial_1000':
        """
        For this implementation to work you'll need to have the videos loaded into a directory under
        '../data/videos'
        """
 
        sys.path.append("..")

        top_level_path = '../data/videos'
        video_list = os.listdir(top_level_path)

        if addition_parameters is None:
            first_n_videos = len(video_list)
        else:
            if 'first_n_videos' in addition_parameters.keys():
                first_n_videos = addition_parameters['first_n_videos']
            else:
                first_n_videos = len(video_list)

        video_views = get_video_play_count(input_type=input_type)

        x_dir = f"../data/x_tensors/{input_type}/"
        y_dir = f"../data/y_tensors/{input_type}/"

        os.makedirs(x_dir, exist_ok=True)
        os.makedirs(y_dir, exist_ok=True)

        for tiktok_video in video_list[:first_n_videos]:
            video_path = f"{top_level_path}/{tiktok_video}"

            if verbose:
                print(f'Currently processing: {tiktok_video}')

            tiktok_video_id = tiktok_video.split(sep='.')[0]
            vf, af, info, meta = process_video(video_object_path=video_path, device=device)
            
            x_file_path = f"{x_dir}{tiktok_video_id}_x_tensor.pt"
            y_file_path = f"{y_dir}{tiktok_video_id}_y_tensor.pt"

            # Save the x & y tensors
            with open(x_file_path, 'wb') as x_file, open(y_file_path, 'wb') as y_file:
                # Converting target video play count to tensor and adding dimension 
                y_data = torch.as_tensor([video_views[tiktok_video_id]]).float().unsqueeze(1)
                
                torch.save(vf, x_file)
                torch.save(y_data, y_file)

            if verbose:
                print(f'Done processing: {video_path}.')
                print(f'X Tensor ({vf.shape}) saved under: {x_file_path}.')
                print(f'Y Tensor ({y_data.shape}) saved under: {y_file_path}.')
                print(f'metadata:\n{meta}\n\n')

        return
    else:
        raise Exception(f'Enter valid input type: {get_valid_input_types()}')


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
    else:
        raise Exception(f'Retrieval of video play count not implemented for {input_type} dataset.')

    return video_views


def process_video(video_object_path, start=0, end=None, read_video=True, read_audio=False, device='CPU'):
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
    return ['initial_1000']