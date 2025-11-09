import requests
from PIL import Image
from io import BytesIO
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
import requests
from io import BytesIO
import re
import numpy as np
from methods.utils import load_images, string_to_token_ids

from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import LogitsProcessorList


# Get the token embeddings from LLaVA
def get_vocab_embeddings_llava(llm_model, tokenizer, device="cuda"):
    vocab = tokenizer.get_vocab()
    llm_tokens = (
        torch.tensor(list(vocab.values()), dtype=torch.long).unsqueeze(0).to(device)
    )
    token_embeddings = llm_model.get_input_embeddings()(llm_tokens)
    return token_embeddings


# Weaves in the image token placeholders into the provided text prompt
def generate_text_prompt(model, model_name, text_prompt):
    qs = text_prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    return conv


def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd

def generate_images_tensor(model, img_path, image_processor):
    image_files = [img_path]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]

    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    ).unsqueeze(0)


    return images_tensor, images, image_sizes

def generate_images_tensor1(model, img_path, image_processor):
    image_files = [img_path]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]

    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    images_tensor= add_diffusion_noise(images_tensor,500)

    return images_tensor, images, image_sizes


def prompt_to_img_input_ids(prompt, tokenizer):
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    return input_ids


def run_llava_model(
    model,
    model_name,
    images_tensor,
    image_sizes,
    tokenizer,
    text_prompt=None,
    hidden_states=False,
):
    if text_prompt == None:
        text_prompt = "Write a detailed description."
    if text_prompt == ".":
        max_new_tokens = 1
    else:
        max_new_tokens = 1024
    conv = generate_text_prompt(model["model"], model_name, text_prompt)
    input_ids = prompt_to_img_input_ids(conv.get_prompt(), tokenizer)
    #model = model.to
    #import pdb; pdb.set_trace()
    # Model parameters
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output = model["model"].generate(
            input_ids,
            images=images_tensor,
            temperature=1.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            # use_cache=True,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            output_hidden_states=hidden_states,
            return_dict_in_generate=True,
            image_sizes=image_sizes,
        )


    outputs = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[
        0
    ].strip()

    hidden_states = torch.stack(output.hidden_states[0])
    image_token_index = input_ids.tolist()[0].index(-200)
    last_hidden_states = hidden_states[:, :, -1, :]
    hidden_states = hidden_states[:, :, image_token_index : image_token_index + (24 * 24),:]
    cls_token = model["model"].model.vision_tower.cls_token
    cls_token = cls_token.to(dtype=torch.float16).clone()   # break "inference" status
    cls_proj  = model["model"].model.mm_projector(cls_token)
 
    return cls_proj, hidden_states, last_hidden_states, outputs


def retrieve_logit_lens_llava(state, img_path, args, text_prompt = None):
    images_tensor, images, image_sizes = generate_images_tensor(
            state["model"], img_path, state["image_processor"]
    )
    input_ids, output = run_llava_model(
        state["model"],
        state["model_name"],
        images_tensor,
        image_sizes,
        state["tokenizer"],
        hidden_states=True,
        text_prompt=text_prompt
    )

    # output_ids = output.sequences
    # o = state["tokenizer"].batch_decode(
    #     output_ids, skip_special_tokens=True
    # )[0]
    # caption = o.strip()

    hidden_states = torch.stack(output.hidden_states[0])
    image_token_index = input_ids.tolist()[0].index(-200)
    last_hidden_states = hidden_states[:, :, -1, :]
    hidden_states = hidden_states[:, :, image_token_index : image_token_index + (24 * 24),:]
    #last_hidden_states = hidden_states[:, :, -1, :]
    cls_proj = state["model"].model.mm_projector(state["model"].model.vision_tower.cls_token.half())
 
    return cls_proj, hidden_states, last_hidden_states#, softmax_probs


def reshape_llava_prompt_hidden_layers(hidden_states):
    prompt_hidden_states = hidden_states[
        0
    ]  # shape is (# layers, # beams, # prompt tokens, # dim size)
    first_beam_layers = torch.stack(list(prompt_hidden_states), dim=0)[:, 0]
    return first_beam_layers


def get_hidden_text_embedding(
    target_word, model, vocab_embeddings, tokenizer, layer=5, device="cuda"
):
    # Tokenize the target word into input ids
    token_ids = string_to_token_ids(target_word, tokenizer)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    # Model parameters
    stop_str = target_word
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            temperature=1.0,
            num_beams=5,
            max_new_tokens=10,  # can be small because we only care about image representations
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=False,
            stopping_criteria=[stopping_criteria],
        )

    hidden_states = reshape_llava_prompt_hidden_layers(output["hidden_states"])

    dist = torch.norm(
        hidden_states[0, len(token_ids) - 1]
        - vocab_embeddings[0, token_ids[len(token_ids) - 1]]
    )
    if dist > 0.1:
        print(
            f"Validation check failed: caption word {target_word} didn't match: {dist}"
        )

    return hidden_states[layer, len(token_ids) - 1].unsqueeze(0)


def get_caption_from_llava(
    img_path, model, model_name, tokenizer, image_processor, text_prompt=None
):
    images_tensor, images, image_sizes = generate_images_tensor(
        model, img_path, image_processor
    )

    # Generate the new caption
    new_caption = run_llava_model(
        model,
        model_name,
        images_tensor,
        image_sizes,
        tokenizer,
        text_prompt=text_prompt,
    )

    return new_caption


def load_llava_state(rank):
    model_path = "liuhaotian/llava-v1.5-13b"
    model_name = get_model_name_from_path(model_path)

    # Don't use device=..., pass directly to map loading
    device_str = f"cuda:{rank}"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name,  device_map=device_str, torch_dtype=torch.float16
    )

    vocabulary = tokenizer.get_vocab()
    vocab_embeddings = get_vocab_embeddings_llava(model, tokenizer, device=f"cuda:{rank}")

    execute_model = lambda img_path, text_prompt=None, image_embeddings=None: get_caption_from_llava(
        img_path, model, model_name, tokenizer, image_processor, text_prompt=text_prompt
    )

    register_hook = (
        lambda hook, layer: model.get_model().layers[layer].register_forward_hook(hook)
    )
    register_pre_hook = (
        lambda pre_hook, layer: model.get_model().layers[layer].register_forward_pre_hook(pre_hook)
    )

    hidden_layer_embedding = lambda text, layer: get_hidden_text_embedding(
        text, model, vocab_embeddings, tokenizer, layer, device=f"cuda:{rank}"
    )

    return {
        "vocabulary": vocabulary,
        "vocab_embeddings": vocab_embeddings,
        "tokenizer": tokenizer,
        "execute_model": execute_model,
        "register_hook": register_hook,
        "register_pre_hook": register_pre_hook,
        "hidden_layer_embedding": hidden_layer_embedding,
        "model": model,
        "model_name": model_name,
        "image_processor": image_processor,
    }
