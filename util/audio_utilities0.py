import subprocess
import whisper
import os
import torch
from pathlib import Path
from whisper import Whisper
from tqdm import tqdm

# Extracts audio into wav format 
def extract_audio(video_folder_path: str, output_dir: str):
    print('Extract in audio from video pack to .wav format..')
    # already_extracted_audios = os.listdir(output_dir)
    for video_file in tqdm(os.listdir(video_folder_path)):
    #     if audio_exists(video_file, already_extracted_audios):
    #         continue

        video_path = os.path.join(video_folder_path, video_file)
        video_id = video_path.split('/')[-1].split('.')[0]
        output_path = f'{output_dir}{video_id}.wav'
        
        args = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_path,
            '-y'
        ]

        proc = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        proc.check_returncode()

# Extracts embeddings and saves in output dir
def extract_embeddings(audio_file_path: str, output_dir: str):
    # track embeddings already extracted
    already_extracted_embeddings = os.listdir(output_dir)

    model = whisper.load_model('base', device='cuda')
    print('Converting to audio files to embeddings..')
    for audio_file in tqdm(os.listdir(audio_file_path)):
        if embedding_exists(audio_file, already_extracted_embeddings):
            continue

        audio_path = os.path.join(audio_file_path, audio_file)
        id = audio_path.split('/')[-1].split('.')[0]
        output_path = f'{output_dir}{id}.pt'

        # issue extracting embeddings for certain videos; catch and continue
        try:
            result = whisper.transcribe(model=model, audio=audio_path)
            segment = result['segments']
            all_embeddings = []
            for i in segment:
                encoder_embeddings = torch.from_numpy(i['encoder_embeddings'])
                all_embeddings.append(encoder_embeddings)
            
            torch.save(all_embeddings, output_path)
        except Exception as error:
            print(f'Error extracting embeddings for {id}.')
            print(error)


def embedding_exists(audio_file: str, already_extracted_embeddings: list):
    embedding_file_name = f'{audio_file.split(".")[0]}.pt'
    if embedding_file_name in already_extracted_embeddings:
        return True
    else:
        return False
    
def audio_exists(video_file: str, already_extracted_audios: list):
    audio_file_name = f'{video_file.split(".")[0]}.wav'
    if audio_file_name in already_extracted_audios:
        return True
    else:
        return False