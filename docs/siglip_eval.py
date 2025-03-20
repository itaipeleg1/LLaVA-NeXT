import os
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

from pathlib import Path
import pandas as pd
import torch.nn.functional as F
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.language_model.llava_qwen import LlavaQwenConfig
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import os
from PIL import Image
import requests
import copy
import torch


def prepare_llava_model():
    # pretrained = "lmms-lab/llava-onevision-qwen2-7b-si"
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": "sdpa",
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                          device_map=device_map,
                                                                          **llava_model_args)  # Add any other thing you want to pass in llava_model_args

    model.eval()
    return tokenizer, model, image_processor, max_length


def save_cls(image_path, cls, dirname="llm_cls"):
    # save the cls tensor to a file
    # change the dir
    save_path = Path(image_path).parent.parent / dirname
    save_path.mkdir(exist_ok=True)
    cls_path = save_path / image_path.with_suffix(".pt").name
    torch.save(cls, cls_path)


def load_cls(pt_path):
    # load the cls tensor from a file
    cls = torch.load(pt_path)
    return cls

def get_vision_embedding(model, tokenizer, image_processor, device, image_path):
    path = r"D:\Projects\Annotators\data\png\49408.png"
    path = path.replace("\\", os.sep)
    path = path.replace("D:", '/mnt/d')
    # image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\nAnswer in Yes or No. Are the people in the foreground in a social interaction?"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)
    image_sizes = [image.size]

    kwargs = {'do_sample': False, 'max_new_tokens': 4096, 'temperature': 0}
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    modalities = ["image"]
    if image_tensor is not None:
        (inputs, position_ids, attention_mask, _, inputs_embeds, _, image_embeds, user_text_embeds) = model.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, None, None, image_tensor, modalities, image_sizes=image_sizes)
    # model.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, image_tensor, modalities, image_sizes=image_sizes)
    #
    # # Now lets do global average pooling on the image embeddings
    image_embeds = image_embeds[0]
    image_embeds = image_embeds.mean(dim=0)
    text_embeds = user_text_embeds.mean(dim=0)
    cos_sim = F.cosine_similarity(image_embeds.unsqueeze(0), text_embeds.unsqueeze(0))
    print(f"Cosine similarity between image and text embeddings: {cos_sim}")
    # todo: tokenize & embed several prompts, and test the image vs text embeddings alignment in the latent space via cosine similarity

def generate_full_conversation(model, tokenizer, image_processor,prompt, device, path):
    path = path.replace("\\", os.sep)
    path = path.replace("D:", '/mnt/d')
    # image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)
    image_sizes = [image.size]

    kwargs = {'do_sample': False, 'max_new_tokens': 4096, 'temperature': 0}
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")
    cont, image_embeds = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    cls = cont.hidden_states[-1][-1].view(-1)
    text_outputs = tokenizer.batch_decode(cont.sequences[0], skip_special_tokens=True)
    del cont
    return text_outputs, image_embeds, cls

# def load_pretrained_siglip(model_path):
#     config = LlavaQwenConfig.from_pretrained(model_path)
#     vision_tower = build_vision_tower(config, delay_load=False)
    #image_features = [x.flatten(0, 1) for x in image_features]
    #base_image_feature = image_feature[0]
    # if "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
    # unit = image_feature.shape[2]
    # image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    # image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    # image_feature = unpad_image(image_feature, image_sizes[image_idx])
    # c, h, w = image_feature.shape
    # times = math.sqrt(h * w / (max_num_patches * unit ** 2))
    # if times > 1.1:
    #     image_feature = image_feature[None]
    #     image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
    # image_feature = torch.cat((image_feature,
    #                            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(
    #                                image_feature.device)), dim=-1)
    # image_feature = image_feature.flatten(1, 2).transpose(0, 1)


    #LlavaQwenForCausalLM:
    # position_ids = kwargs.pop("position_ids", None)
    # attention_mask = kwargs.pop("attention_mask", None)
    # if "inputs_embeds" in kwargs:
    #     raise NotImplementedError("`inputs_embeds` is not supported")
    #
    # if images is not None:
    #     (inputs, position_ids, attention_mask, _, inputs_embeds, _, _) = self.prepare_inputs_labels_for_multimodal(
    #         inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
    #LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)

def social_annot_comparison_routine():
    device = "cuda"
    annot_df_path = r"D:\Projects\Annotators\data\frame_distances_avg_1s_full.csv"
    png_dir = r"D:\Projects\Annotators\data\png"
    if 'psychology' in hostname:
        png_dir = r"/home/new_storage/experiments/NND/500daysofsummer/png"
        annot_df_path = "/home/Alon/data/summer/annotations/frame_distances_avg_1s_full.csv"
    else:
        png_dir = png_dir.replace("\\", os.sep)
        png_dir = png_dir.replace("D:", '/mnt/d')
        annot_df_path = annot_df_path.replace("\\", os.sep)
        annot_df_path = annot_df_path.replace("D:", '/mnt/d')
    results_path = annot_df_path.replace(".csv", "_llava_answers.csv")
    annot_df = pd.read_csv(annot_df_path)
    # skip the initial offset frames
    offset = 40
    annot_df = annot_df[offset:]
    #Downsample the dataframe by 3 since it's originally annotated at every 3rd frame
    # annot_df = annot_df[::3]
    tokenizer, model, image_processor, max_length = prepare_llava_model()
    nb_processed_images = 0
    for idx, row in annot_df.iterrows():
        image_path = Path(png_dir) / f"{row['frame']}.png"
        if not image_path.exists():
            continue
        prompt = "Answer with 'Yes' or 'No' and provide a brief explanation. Are the people in the foreground engaged in a social interaction?"
        answer, image_embeds, cls = generate_full_conversation(model, tokenizer, image_processor, prompt, device, str(image_path))

        save_cls(image_path, cls)

        print(f"Frame: {row['frame']}, GT:{row['gt_social_nonsocial']}, Answer: {answer[0]}")
        # save the generated answer to the dataframe
        annot_df.loc[idx, 'question'] = prompt
        annot_df.loc[idx, 'llava_answer'] = answer[0]
        nb_processed_images += 1
        if nb_processed_images % 30 == 0:
            annot_df.to_csv(results_path, index=False)
            print(f"Saved results to {results_path}")

def ucolaeo_comparison_routine():
    device = "cuda"
    png_dir = r"D:\Projects\data\ucolaeodb\frames"
    png_dir = png_dir.replace("\\", os.sep)
    png_dir = png_dir.replace("D:", '/mnt/d')
    png_dir = r"/home/Alon/data/summer/png"
    results_path = png_dir.replace("frames", "ucolaeo_llava_answers.csv")

    tokenizer, model, image_processor, max_length = prepare_llava_model()
    results_df = pd.DataFrame(columns=['dirname', 'frame_name','question1', 'answer1', 'question2', 'answer2'])
    answer1 = ['']
    answer2 = ['']
    nb_processed_images = 0
    # run through the png dir with os walk
    for root, dirs, files in os.walk(png_dir):
        for file in files:
            image_path = Path(root) / file
            # image_path = Path(r"D:\Projects\data\ucolaeodb\frames\mr19\000016.jpg")
            # set filename to be the file + upper directory name
            img_dir = Path(root).parts[-1]
            filename = image_path.stem
            prompt1 = "Answer in Yes or No. Are both people looking at each other?"
            answer1 = generate_full_conversation(model, tokenizer, image_processor, prompt1, device, str(image_path))
            prompt2 = "Answer in Yes or No. Is there at least one person looking at another?"
            # answer2 = generate_full_conversation(model, tokenizer, image_processor, prompt2, device, str(image_path))
            # prompt2 = "Answer in Yes or No. Are there people looking at the same object?"
            # answer2 = generate_full_conversation(model, tokenizer, image_processor, prompt2, device, str(image_path))
            dirname = Path(root).parts[-1]
            print(f"Dirname:{dirname}, Frame: {filename}, Answer1: {answer1[0]}, Answer2: {answer2[0]}")
            single_result_df = pd.DataFrame([[dirname, filename, prompt1, answer1[0], prompt2, answer2[0]]], columns=['dirname', 'frame_name', 'question1', 'answer1', 'question2', 'answer2'])
            results_df = pd.concat([results_df, single_result_df], ignore_index=True)
            nb_processed_images += 1
            if nb_processed_images % 30 == 0:
                print(f"Processed {nb_processed_images} images")
                results_df.to_csv(results_path, index=False)
                print(f"Saved results to {results_path}")


if __name__ == "__main__":
    # ucolaeo_comparison_routine()
    social_annot_comparison_routine()


