import os
from operator import attrgetter
from pathlib import Path

import pandas as pd

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from docs.clip_eval import fix_wsl_paths, save_cls
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

def get_hostname():
    import socket
    hostname = socket.gethostname()
    print("Running on:", hostname)
    return hostname
hostname = get_hostname()
if 'psychology' in hostname:
    os.environ["HF_HOME"] = "/home/new_storage/HuggingFace_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/home/new_storage/HuggingFace_cache"
    os.environ["HF_DATASETS_CACHE"] = "/home/new_storage/HuggingFace_cache"
    os.environ["HF_TOKENIZERS_CACHE"] = "/home/new_storage/HuggingFace_cache"

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()


# Function to extract frames from video
def load_movie(video_path, max_frames_num, output_dir, vr=None, start_frame=0, fps=25, duration=3, **kwargs):
    if vr is None:
        vr = VideoReader(video_path, ctx=cpu(0))
    end_frame = start_frame + fps * duration
    uniform_sampled_frames = np.linspace(start_frame, end_frame, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    res = {'frames': spare_frames, 'start_frame': start_frame, 'end_frame': end_frame, 'vr': vr}
    # save the video in the output directory
    output_path = os.path.join(output_dir, f"{start_frame}_{end_frame}.mp4")
    # create_video_clip(spare_frames, output_path)

    return res

# create and save compressed video clip
def create_video_clip(frames, output_path):
    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 25, (frames.shape[2], frames.shape[1]))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Video clip saved at {output_path}")

    return output_path


# Load and process video
if 'psychology' in hostname:
    # png_dir = r"/home/new_storage/experiments/NND/500daysofsummer/png"
    # annot_df_path = "/home/Alon/data/summer/annotations/frame_distances_avg_1s_full.csv"
    video_path = r"/home/new_storage/experiments/NND/500 Days Of Summer.2009.720p.BDRip.x264-VLiS.mp4"
else:
    video_path = r"D:\Projects\Annotators\data\Sherlock.S01E01.A.Study.in.Pink.mkv"
    # video_path = r"E:\moments\Moments_in_Time_layla\training\working\t2z1X76v4ys_178.mp4"
    video_path = video_path.replace("\\", os.sep)
    drive = video_path.split(os.sep)[0]
    video_path = video_path.replace(drive, f'/mnt/{drive[0].lower()}')
output_dir = Path(video_path).parent / 'Sherlock'
output_dir.mkdir(exist_ok=True, parents=True)
result_path = output_dir / 'llava_3s_video_results_primitives.csv'
result_df = pd.DataFrame(columns=['seconds', 'start_frame', 'end_frame', 'question', 'response'])
fps = 25
offset = 243 * fps
# offset = 0
vr = VideoReader(video_path, ctx=cpu(0))

for frame_ind in range(offset, len(vr), fps * 3):
    video_frames_dd = {"frames": None, "start_frame": frame_ind, "vr": vr, "max_frames_num": 16,
                       "video_path": video_path, "fps": fps, "duration": 3, "output_dir": output_dir}
    video_frames_dd = load_movie(**video_frames_dd)
    video_frames = video_frames_dd['frames']

    # print(video_frames.shape) # (16, 1024, 576, 3)
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)

    # Prepare conversation input
    conv_template = "qwen_1_5"
    question = (f"Describe where each of people is located in the video, what are they doing, and where are they looking")
    full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n {question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont, image_embeds = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_hidden_states=True,
        return_dict_in_generate=True,
        modalities=["video"],
    )
    text_outputs = tokenizer.batch_decode(cont.sequences, skip_special_tokens=True)[0]
    frame_time = frame_ind // fps
    # convert to minutes : seconds
    minutes = frame_time // 60
    seconds = frame_time % 60
    print(f"Frame: {frame_time} ({minutes}:{seconds})")
    print(text_outputs)
    language_latent = cont.hidden_states[-1][-1].view(-1)
    save_cls(str(frame_ind), language_latent, output_dir=output_dir / 'llm_language_embeds')
    save_cls(str(frame_ind), image_embeds[0], output_dir=output_dir / 'vision_only_embeds')

    clip_res = {'start_frame': video_frames_dd['start_frame'], 'end_frame': video_frames_dd['end_frame'],
                'seconds': frame_ind // fps,  'question': question, 'response': text_outputs[0]}
    clip_df = pd.DataFrame(clip_res, index=[0])
    result_df = pd.concat([result_df, clip_df], ignore_index=True)
    # save every 10 responses
    if len(result_df) % 10 == 0:
        result_df.to_csv(result_path, index=False)
