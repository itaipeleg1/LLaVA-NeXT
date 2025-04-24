import os
from operator import attrgetter
from pathlib import Path
import math
import pandas as pd

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from clip_eval import fix_wsl_paths, save_cls
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

os.environ["HF_HOME"] = "/home/new_storage/sherlock/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/new_storage/sherlock/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/home/new_storage/sherlock/hf_cache"
os.environ["HF_TOKENIZERS_CACHE"] = "/home/new_storage/sherlock/hf_cache"


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
video_path ="/home/new_storage/sherlock/STS_sherlock/projects data/Sherlock.S01E01.A.Study.in.Pink.1080p.10bit.BluRay.5.1.x265.HEVC-MZABI.mkv"
output_dir = Path(video_path).parent / 'Sherlock'
output_dir.mkdir(exist_ok=True, parents=True)

result_df = pd.DataFrame(columns=['seconds', 'start_frame', 'end_frame', 'question', 'response'])
fps = 25
durs = np.arange(6, 15, 1.5) # 3 seconds
#offset = 243 * fps
offset = 0
vr = VideoReader(video_path, ctx=cpu(0))
# Get the total number of frames in the video
for duration in durs:

    print(f"***********************************Processing video with duration: {duration} seconds*************************************")
    step_float = duration * fps
    use_alternating = not step_float.is_integer()
    ## reset the df
    result_df = pd.DataFrame(columns=['seconds', 'start_frame', 'end_frame', 'question', 'response'])
    if use_alternating:
        steps = [math.floor(step_float), math.ceil(step_float)]
        step_idx = 0
    else:
        step_size = int(step_float)

    frame_ind = offset
    max_frame_limit  = 71250
    while frame_ind + int(fps * duration) <= max_frame_limit:
        # Process clip...
        start_frame = frame_ind
        end_frame = start_frame + int(fps * duration)
        target_fps = 6 
        max_frames_num = int(duration * target_fps)
        max_frames_num = min(max_frames_num, 30)
        video_frames_dd = {
            "frames": None,
            "start_frame": frame_ind,
            "vr": vr,
            "max_frames_num": max_frames_num,
            "video_path": video_path,
            "fps": fps,
            "duration": duration,
            "output_dir": output_dir
        }
        result_path = output_dir / f'llava_{duration}s_video_resultsnew.csv'
        video_frames_dd = load_movie(**video_frames_dd)
        video_frames = video_frames_dd["frames"]

        # print(video_frames.shape) # (16, 1024, 576, 3)
        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        # Prepare conversation input
        conv_template = "qwen_1_5"
        question = (f"Does this video contain social interaction?")
        full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n {question}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [frame.size for frame in video_frames]

        # Generate response
        cont = model.generate(
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
        # DEBUG: Sanity check current frame info
        print(f"➡️  Processing frame_ind={frame_ind}, time={minutes:02}:{seconds:02}, end_frame={end_frame}")

        # Save CLS token
        language_latent = cont.hidden_states[-1][-1].view(-1)
        duration_str = str(duration).replace(".", "_")
        save_cls(str(frame_ind), language_latent, output_dir=output_dir / f'llm_language_embedsTR{duration_str}new')

        # Append result
        clip_res = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'seconds': frame_time,
            'question': question,
            'response': text_outputs[0]  
        }
        clip_df = pd.DataFrame(clip_res, index=[0])
        result_df = pd.concat([result_df, clip_df], ignore_index=True)

        # Periodically save
        if len(result_df) % 10 == 0:
            result_df.to_csv(result_path, index=False)

        # Stepping control
        if use_alternating:
            step = steps[step_idx % 2]
            step_idx += 1
        else:
            step = step_size

        print(f"🔁 Advancing frame_ind by {step} frames, response: {text_outputs}")
        frame_ind += step
