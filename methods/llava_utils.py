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


def compute_binary_conditional_likelihood_llava(
    model,
    model_name,
    images_tensor,
    image_sizes,
    tokenizer,
    prefix_prompt,  # e.g., "Select the correct interaction from the list:"
    target_text,   # List of strings, e.g., ["a photo of a person kicking a ball", ...]
):
    # Generate conversation template with prefix
    conv = generate_text_prompt(model["model"], model_name, prefix_prompt)
    prefix_with_image = conv.get_prompt()
    
    # Tokenize prefix (contains image token placeholder -200)
    prefix_ids = prompt_to_img_input_ids(prefix_with_image, tokenizer)
    
    log_probs = []
    
    with torch.inference_mode():
        # Forward pass to get logits
        #import pdb; pdb.set_trace()
        outputs = model["model"](
            input_ids=prefix_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            return_dict=True,
            output_attentions=True
        )
        
        # Logits shape: [batch_size, seq_len, vocab_size]
        logits = outputs.logits


    next_token_logits = logits[:, -1, :]  # [1, vocab_size]
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids
    no_id  = tokenizer("No",  add_special_tokens=False).input_ids

    yes_logp = log_probs[0, yes_id]
    no_logp  = log_probs[0, no_id]

    yes_prob = yes_logp.exp().item()
    no_prob  = no_logp.exp().item()
    
    return yes_prob, no_prob


def compute_conditional_likelihood_llava(
    model,
    model_name,
    images_tensor,
    image_sizes,
    tokenizer,
    prefix_prompt,  # e.g., "Select the correct interaction from the list:"
    target_text,   # List of strings, e.g., ["a photo of a person kicking a ball", ...]
):
    """
    Compute the conditional log-likelihood of target texts given an image and prefix.
    
    Args:
        model: LLaVA model dict with 'model' key
        model_name: Name of the model
        images_tensor: Preprocessed image tensor
        image_sizes: Image sizes
        tokenizer: Tokenizer
        prefix_prompt: The question/prompt text (image token will be auto-inserted)
        target_texts: List of candidate answer strings to score
    
    Returns:
        log_probs: List of average log probabilities for each target text
        probs: List of probabilities (exponentiated log probs)
    """
    # Generate conversation template with prefix
    conv = generate_text_prompt(model["model"], model_name, prefix_prompt)
    prefix_with_image = conv.get_prompt()
    
    # Tokenize prefix (contains image token placeholder -200)
    prefix_ids = prompt_to_img_input_ids(prefix_with_image, tokenizer)
    
    log_probs = []
    
    with torch.inference_mode():
        # Tokenize target text (continuation after prefix)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        target_ids = torch.tensor(target_ids, dtype=torch.long, device=prefix_ids.device)
        
        # Concatenate: [prefix with <image>] + [target text]
        full_input_ids = torch.cat([prefix_ids, target_ids.unsqueeze(0)], dim=1)
        #import pdb; pdb.set_trace()
        ids = full_input_ids[0]
        valid = (ids >= 0) & (ids < tokenizer.vocab_size)
        # Forward pass to get logits
        #import pdb; pdb.set_trace()
        outputs = model["model"](
            input_ids=full_input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            return_dict=True,
            output_hidden_states=True
        )
        
        # Logits shape: [batch_size, seq_len, vocab_size]
        logits = outputs.logits


        hidden_states = torch.stack(outputs.hidden_states)
        image_token_index = full_input_ids.tolist()[0].index(-200)
       
        visual_hidden_states = hidden_states[:, :, image_token_index : image_token_index + (24 * 24),:]

        # Get logits that predict target tokens

        # image_token_index = full_input_ids.tolist()[0].index(-200)
        # num_image_tokens = (prefix_ids == image_token_index).sum().item()
        image_patch_tokens = 576  # 24x24 patches
        actual_prefix_len = prefix_ids.shape[1] - 1 + 576
        target_len = len(target_ids)
        target_logits = logits[0, actual_prefix_len-1:actual_prefix_len-1+target_len, :]
        #import pdb; pdb.set_trace()
        # Compute log softmax
        tgt_hidden_states = hidden_states[:, :, actual_prefix_len-1:actual_prefix_len-1+target_len, :]
        log_probs_dist = torch.nn.functional.log_softmax(target_logits, dim=-1)
        
        # Extract log prob of actual target tokens
        token_log_probs = log_probs_dist[range(target_len), target_ids]
        #import pdb; pdb.set_trace()
        # Average log probability across tokens
        # Perplexity (lower is better)
        #import pdb; pdb.set_trace()
        # perplexity = torch.exp(-token_log_probs.mean()).item()
        avg_log_prob = token_log_probs.mean().item()
        log_probs.append(avg_log_prob)

    # Convert to probabilities
    probs = [np.exp(lp) for lp in log_probs]
    
    return log_probs, probs, visual_hidden_states, tgt_hidden_states


# Example usage:
# target_texts = [
#     "a photo of a person kicking a ball",
#     "a photo of a person holding a cup",
#     "a photo of a person riding a bike"
# ]
# log_probs, probs = compute_conditional_likelihood_llava(
#     model, model_name, images_tensor, image_sizes, tokenizer,
#     prefix_prompt="Select the correct interaction from the list:",
#     target_texts=target_texts
# )
# best_idx = np.argmax(probs)
# print(f"Best match: {target_texts[best_idx]} with prob {probs[best_idx]:.4f}")


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



def generate_images_tensor(model, img_path, image_processor):
    image_files = [img_path]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]

    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    ).unsqueeze(0)


    return images_tensor, images, image_sizes


def prompt_to_img_input_ids(prompt, tokenizer):
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    return input_ids
    
def get_img_idx(model, model_name, tokenizer, text_prompt):
    conv = generate_text_prompt(model["model"], model_name, text_prompt)
    input_ids = prompt_to_img_input_ids(conv.get_prompt(), tokenizer)
    image_patch_tokens = 576  # 24x24 patches
    actual_prefix_len = input_ids.shape[1] - 1 + 576
    return input_ids.tolist()[0].index(-200), input_ids.tolist()[0].index(-200) + 576, actual_prefix_len

def run_llava_model(
    model,
    model_name,
    images_tensor,
    image_sizes,
    tokenizer,
    text_prompt=None,
    hidden_states=False,
):
    if text_prompt is None:
        text_prompt = "Write a detailed description."

    conv = generate_text_prompt(model["model"], model_name, text_prompt)
    input_ids = prompt_to_img_input_ids(conv.get_prompt(), tokenizer)

    output = model["model"](
        input_ids=input_ids,
        images=images_tensor,
        image_sizes=[image_sizes],
        output_hidden_states=True,
        return_dict=True,
    )

    # output.hidden_states: tuple of (num_layers+1) x [1, seq_len, dim]
    hidden = torch.stack(output.hidden_states)  # [layers+1, 1, seq_len, dim]
    image_token_index = input_ids.tolist()[0].index(-200)
    img_hidden_states = hidden[:, :, image_token_index : image_token_index + (24 * 24), :]

    return img_hidden_states, output, None, None


def generate_llava_model(
    model,
    model_name,
    images_tensor,
    image_sizes,
    tokenizer,
    text_prompt=None,
    hidden_states=False,
):
    if text_prompt is None:
        text_prompt = "Write a detailed description."

    has_image = images_tensor is not None

    if has_image:
        conv = generate_text_prompt(model["model"], model_name, text_prompt)
        input_ids = prompt_to_img_input_ids(conv.get_prompt(), tokenizer)
    else:
        # Text-only: build conv without image token
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text_prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = (
            tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.cuda()
        )

    with torch.inference_mode():
        generate_kwargs = dict(
            inputs=input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
        )
        if has_image:
            generate_kwargs["images"] = images_tensor
            generate_kwargs["image_sizes"] = [image_sizes]

        output = model["model"].generate(**generate_kwargs)

    hidden = torch.stack(output.hidden_states[0])  # [layers+1, 1, seq_len, dim]

    if has_image:
        image_token_index = input_ids.tolist()[0].index(-200)
        img_hidden_states = hidden[:, :, image_token_index : image_token_index + (24 * 24), :]
    else:
        img_hidden_states = None

    generated_ids = output.sequences
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return img_hidden_states, generated_text, generated_ids, torch.stack(output.scores, dim=0).squeeze(1)


def retrieve_logit_lens_llava(state, img_path, args, text_prompt = None):
    images_tensor, images, image_sizes = generate_images_tensor(
            state["model"], img_path, state["image_processor"]
    )
    input_ids, output, output_sequnce = run_llava_model(
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
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)

    # Don't use device=..., pass directly to map loading
    device_str = f"cuda:{rank}"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name,  device_map=device_str, torch_dtype=torch.bfloat16 ,use_flash_attn=True
    )
    # Convert entire model (including vision encoder & mm_projector) to bfloat16
    model = model.to(torch.bfloat16)

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



import torch

def get_device_from_module(module):
    return next(module.parameters()).device

def get_phrase_embedding(phrase, vocab_embeddings, tokenizer, remove_first=True):
    # returns size (1, 5120)
    text_embeddings = []
    for token_id in tokenizer(phrase)["input_ids"]:
        text_embeddings.append(vocab_embeddings[:, token_id])
    if remove_first:
        text_embeddings = text_embeddings[1:]
    phrase_embedding = torch.sum(
        torch.concat(text_embeddings), dim=0, keepdim=True
    ) / len(text_embeddings)
    return phrase_embedding


def projection(image_embeddings, text_embedding):
    return (image_embeddings @ text_embedding.T)[0, :, 0] / (
        text_embedding @ text_embedding.T
    ).squeeze()


def subtract_projection(image_embeddings, text_embedding, weight=1, device = None):
       # if device is None, don't move the embeddings to any device - keep their pre-existing device configs in tact
    if device != None:
        image_embeddings = image_embeddings.to(device)
        text_embedding = text_embedding.to(device)
    image_embeddings = image_embeddings.clone()
    proj = projection(image_embeddings, text_embedding)
    for i in range(image_embeddings.shape[1]):
        if proj[i] > 0:
            image_embeddings[:, i] += weight * proj[i] * text_embedding
    return image_embeddings


def subtract_projections(image_embeddings, text_embeddings, weight=1, device = None):
    # text_embeddings: (# embeds, 1, # dim size)
    # if device is None, don't move the embeddings to any device - keep their pre-existing device configs in tact
    img_embeddings = image_embeddings.clone()
    for text_embedding in text_embeddings:
        img_embeddings = subtract_projection(img_embeddings, text_embedding, weight, device=device)
    return img_embeddings




def remove_all_hooks(model):
    # Iterate over all modules in the model
    for module in model.modules():
        # Clear forward hooks
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()
        # Clear backward hooks (if any)
        if hasattr(module, "_backward_hooks"):
            module._backward_hooks.clear()
        # Clear forward pre-hooks (if any)
        if hasattr(module, "_forward_pre_hooks"):
            module._forward_pre_hooks.clear()


def generate_mass_edit_hook(
    text_embeddings, start_edit_index, end_edit_index, layer, weight=1, minimum_size=32
):
    if len(text_embeddings) == 0:
        print("No text embeddings found. Note that no editing will occur.")
    def edit_embeddings(module, input, output):
        device = get_device_from_module(module)
        new_output = list(output)
        if new_output[0].shape[1] > minimum_size:
            print(f"Editing layer {layer}")
            new_output[0][:, start_edit_index:end_edit_index] = subtract_projections(
                new_output[0][:, start_edit_index:end_edit_index],
                text_embeddings,
                weight=weight,
                device=device
            )
        return tuple(new_output)

    return edit_embeddings


def generate_mass_edit_pre_hook(
    text_embeddings, start_edit_index, end_edit_index, layer, weight=1, minimum_size=32
):
    if len(text_embeddings) == 0:
        print("No text embeddings found. Note that no editing will occur.")
    def edit_embeddings(module, input):
        device = get_device_from_module(module)
        new_input = list(input)
        if new_input[0].shape[1] > minimum_size:
            print(f"Editing layer {layer}")
            new_input[0][:, start_edit_index:end_edit_index] = subtract_projections(
                new_input[0][:, start_edit_index:end_edit_index],
                text_embeddings,
                weight=weight,
                device=device
            )
        return tuple(new_input)

    return edit_embeddings


def internal_confidence(tokenizer, softmax_probs, class_):
    class_token_indices = tokenizer.encode(class_)[1:]
    return softmax_probs[class_token_indices].max()


def internal_confidence_heatmap(tokenizer, softmax_probs, class_):
    class_token_indices = tokenizer.encode(class_)[1:]
    return softmax_probs[class_token_indices].max(axis=0).T


def internal_confidence_segmentation(tokenizer, softmax_probs, class_, num_patches=24):
    class_token_indices = tokenizer.encode(class_)[1:]
    return (
        softmax_probs[class_token_indices]
        .max(axis=0)
        .max(axis=0)
        .reshape(num_patches, num_patches)
        .astype(float)
    )