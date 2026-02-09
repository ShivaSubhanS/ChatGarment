"""
Kaggle-specific inference script for ChatGarment
Optimized for GPU execution on Kaggle with dataset paths
"""
import argparse
import copy
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import subprocess
import random
import pickle as pkl
import transformers
import tokenizers

# Add paths for Kaggle environment
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from functools import partial
from easydict import EasyDict as edict
from typing import Dict, Optional, Sequence, List

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import tqdm
import shutil
from llava.json_fixer import repair_json

# Use Kaggle-specific inference args
from llava.inference_args_kaggle import ModelArguments, DataArguments, TrainingArguments, rank0_print, get_checkpoint_path
from llava.garment_utils_v2 import run_garmentcode_parser_float50

import json
from tqdm import tqdm
import re 

os.environ["MASTER_PORT"] = "23499"


def find_all_linear_names(model, lora_target_modules=['q_proj', 'v_proj']):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                [
                    x not in name
                    for x in [
                        'mm_projector', 'vision_tower', 'vision_resampler', 'float_layer'
                    ]
                ]
            )
            and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))



class LazyImageDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, imagefolder: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 max_len=-1):
        super(LazyImageDataset, self).__init__()
        self.imagefolder = imagefolder
        all_images = [item for item in os.listdir(imagefolder) \
                      if (item.endswith('.png') or item.endswith('.jpg'))]

        self.tokenizer = tokenizer
        self.all_images = all_images
        self.data_args = data_args

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        image_file = os.path.join(self.imagefolder, self.all_images[i])
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        data_dict = {}
        data_dict['image'] = image
        data_dict['image_path'] = os.path.join(image_folder, image_file)

        return data_dict




def translate_args(model_args, data_args, training_args):
    args = edict(
        local_rank=local_rank,
        version=None,
        vis_save_path="./vis_output",
        precision="bf16",
        image_size=None,
        model_max_length=training_args.model_max_length,
        lora_r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        lora_target_modules=None,
        vision_tower=model_args.vision_tower,
        load_in_8bit=False,
        load_in_4bit=False,
        dataset=None,
        sample_rates=None,
        log_base_dir='./runs',
        exp_name="try_lr1e_4_generator_wildimg",
        epochs=40,
        steps_per_epoch=500,
        batch_size=4,
        grad_accumulation_steps=8,
        val_batch_size=1,
        workers=4,
        lr=1e-4,
        ce_loss_weight=1.0,
        no_eval=False,
        eval_only=False,
        vision_pretrained=None,
        resume="",
        start_epoch=0,
        print_freq=1,
        gradient_checkpointing=training_args.gradient_checkpointing,
        beta1=0.9,
        beta2=0.999,

        use_mm_start_end=False
    )   

    return args

    
def main(args):
    print("\n" + "="*80)
    print("Starting ChatGarment Inference on Kaggle GPU")
    print("="*80 + "\n")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ WARNING: No GPU detected! Using CPU (will be slow)")
    
    attn_implementation = 'eager'  # Changed from 'flash_attention_2' to 'eager'
    global local_rank

    print("\nStep 1: Parsing command line arguments...")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    print(f"   - Local rank: {local_rank}")
    print(f"   - Compute dtype: {compute_dtype}")
    print(f"   - LoRA r: {training_args.lora_r}, alpha: {training_args.lora_alpha}")
    print(f"   - Device: {training_args.device}")

    print("\nStep 2: Translating arguments...")
    args = translate_args(model_args, data_args, training_args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    bnb_model_from_pretrained_args = {}
    # writer = None
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    assert training_args.bits not in [4, 8]
    assert model_args.vision_tower is not None
    assert 'mpt' not in model_args.model_name_or_path

    print("\nStep 3: Loading tokenizer...")
    print(f"   - Model: {model_args.model_name_or_path}")
    print(f"   - Cache dir: {training_args.cache_dir}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    print(f"   ✓ Tokenizer loaded, added {num_added_tokens} special tokens")

    print("\nStep 4: Loading base model...")
    print(f"   - Attention implementation: {attn_implementation}")
    model = GarmentGPTFloat50ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        seg_token_idx=args.seg_token_idx,
        # hidden_size=768,
        **bnb_model_from_pretrained_args
    )
    print("   ✓ Base model loaded")
    
    print("\nStep 5: Configuring model...")
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False
    assert not model_args.freeze_backbone

    assert training_args.gradient_checkpointing
    print("   - Enabling gradient checkpointing...")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    assert model_args.version == "v1"
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates: # conv_vicuna_v1
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    print("\nStep 6: Initializing vision modules...")
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    vision_tower = model.get_vision_tower()
    # Move vision tower to GPU
    print("   - Moving vision tower to GPU...")
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device="cuda")
    print("   ✓ Vision tower ready on GPU")
    
    # Move mm_projector to GPU
    print("   - Moving mm_projector to GPU...")
    model.get_model().mm_projector.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device="cuda")
    print("   ✓ mm_projector ready on GPU")

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    print("\nStep 7: Adding LoRA adapters...")
    print(f"   - LoRA r={training_args.lora_r}, alpha={training_args.lora_alpha}")
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    model = get_peft_model(model, lora_config)
    print("   ✓ LoRA adapters added")

    print("\nStep 8: Resizing token embeddings...")
    model.resize_token_embeddings(len(tokenizer))
    print(f"   ✓ Token embeddings resized to {len(tokenizer)}")

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # Copy values from training_args to model_args for llava_arch.py compatibility
    model_args.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter
    model_args.mm_use_im_start_end = training_args.mm_use_im_start_end
    model_args.mm_use_im_patch_token = training_args.mm_use_im_patch_token
    
    # Use training_args for model config
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter
    assert not training_args.tune_mm_mlp_adapter
    
    assert not training_args.freeze_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = training_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = training_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = training_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    # Move model to GPU for Kaggle - Multi-GPU support
    print("\nStep 9: Moving model to GPU...")
    assert args.precision == "bf16"
    model = model.bfloat16()
    
    # Check number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"   - Number of GPUs available: {num_gpus}")
    
    if num_gpus > 1:
        print(f"   - Using DataParallel across {num_gpus} GPUs")
        # Use DataParallel to distribute model across multiple GPUs
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        device = torch.device("cuda")
        print(f"   ✓ Model distributed across {num_gpus} GPUs")
    else:
        model = model.cuda()
        device = torch.device("cuda")
        print(f"   ✓ Model configured for single GPU inference on {device}")
    
    print("\nStep 10: Preparing dataset...")
    print(f"   - Data path: {data_args.data_path_eval}")
    val_dataset = LazyImageDataset(
        tokenizer=tokenizer,
        imagefolder=data_args.data_path_eval,
        data_args=data_args,
    )
    print(f"   ✓ Found {len(val_dataset)} images")

    ########################################################################################
    print("\nStep 11: Loading fine-tuned checkpoint...")
    resume_path = get_checkpoint_path()
    print(f"   - Checkpoint path: {resume_path}")
    state_dict = torch.load(resume_path, map_location="cpu")  # Load to CPU first
    print("   - Loading state dict...")
    
    # Handle vocab size mismatch (32001 in checkpoint vs 32002 in model)
    model_to_load = model.module if num_gpus > 1 else model
    current_state = model_to_load.state_dict()
    
    # Fix embedding size mismatches
    embed_keys = ['base_model.model.model.embed_tokens.weight', 'base_model.model.lm_head.weight']
    for key in embed_keys:
        if key in state_dict and key in current_state:
            ckpt_shape = state_dict[key].shape
            model_shape = current_state[key].shape
            if ckpt_shape != model_shape:
                print(f"   ℹ Resizing {key}: {ckpt_shape} -> {model_shape}")
                # Copy the checkpoint weights for the overlapping vocab
                min_vocab_size = min(ckpt_shape[0], model_shape[0])
                current_state[key][:min_vocab_size] = state_dict[key][:min_vocab_size]
                # Remove from state_dict to avoid loading error
                del state_dict[key]
    
    # Load the rest of the weights
    missing_keys, unexpected_keys = model_to_load.load_state_dict(state_dict, strict=False)
    print("   ✓ Checkpoint loaded successfully")
    
    # Report any mismatches
    if missing_keys:
        # Filter out the embedding keys we already handled
        other_missing = [k for k in missing_keys if k not in embed_keys]
        if other_missing:
            print(f"   ℹ Missing keys: {len(other_missing)} keys")
    if unexpected_keys:
        print(f"   ℹ Unexpected keys (ignored): {len(unexpected_keys)} keys")

    if data_args.data_path_eval[-1] == '/':
        data_args.data_path_eval = data_args.data_path_eval[:-1]
    if data_args.data_path_eval.split('/')[-1] == 'img' or data_args.data_path_eval.split('/')[-1] == 'imgs':
        dataset_name = data_args.data_path_eval.split('/')[-2]
    else:
        dataset_name = data_args.data_path_eval.split('/')[-1]
        
    args.exp_name = resume_path.split('/')[-2]
    parent_folder = os.path.join(args.log_base_dir, args.exp_name, f'{dataset_name}_img_recon')
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    print('val_dataset', len(val_dataset))
    len_val_dataset = len(val_dataset)
    # model.eval()
    
    # Limit to 1 image for testing
    len_val_dataset = min(1, len_val_dataset)
    print(f"\n⚠ LIMITED TO PROCESSING {len_val_dataset} IMAGE(S) FOR TESTING\n")
    
    # hmr_batch = next(iter(train_dataset))
    random.seed(0)
    all_output_dir = []
    all_json_spec_files = []
    
    print("\n" + "="*80)
    print("Starting inference on images...")
    print("="*80 + "\n")
    
    for i in range(len_val_dataset):    
        print(f"\n{'='*60}")
        print(f"DEBUG: Processing image {i+1}/{len_val_dataset}")
        print(f"{'='*60}")
        
        data_item = val_dataset[i]
        image_path = data_item['image_path']
        print(f"DEBUG: Image path: {image_path}")
        
        answers = []
        question1 = 'Can you describe the geometry features of the garments worn by the model in the Json format?'
        question2 = 'Can you estimate the sewing pattern code based on the image and Json format garment geometry description?'
        visualizations = []
        questions = [question1, question2]
        
        for k in range(len(questions)):
            print(f"DEBUG: Processing question {k+1}/2")
            
            conv = conversation_lib.conv_templates[model_args.version].copy()
            conv.messages = []
            if k == 0:
                prompt = question1
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
                
            else:
                # prompt = 'can you describe this pose?'
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + question2 + "\n" + text_output.replace('upper_garment', 'upperbody_garment').replace('lower_garment', 'lowerbody_garment')
                print('DEBUG: Second question prompt:', prompt[:100] + "..." if len(prompt) > 100 else prompt)
                # assert False
            
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_clip = data_item['image']
            print(f"DEBUG: Image tensor shape: {image_clip.shape}, dtype: {image_clip.dtype}")
            
            # Move image to GPU
            image_clip = image_clip.unsqueeze(0).to("cuda")
            assert args.precision == "bf16"
            image_clip = image_clip.bfloat16()
            print(f"DEBUG: Image moved to GPU, shape: {image_clip.shape}, device: {image_clip.device}")
            
            image = image_clip

            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            print(f"DEBUG: Input IDs shape: {input_ids.shape}")
            
            # Move input_ids to GPU
            input_ids = input_ids.unsqueeze(0).to("cuda")
            print(f"DEBUG: Input IDs moved to GPU, shape: {input_ids.shape}, device: {input_ids.device}")

            print("DEBUG: Starting model.evaluate()...")
            # Handle DataParallel wrapper
            model_to_eval = model.module if hasattr(model, 'module') else model
            output_ids, float_preds, seg_token_mask = model_to_eval.evaluate(
                image_clip,
                image,
                input_ids,
                max_new_tokens=2048,
                tokenizer=tokenizer,
            )
            print("DEBUG: Model evaluation completed")

            output_ids = output_ids[0, 1:]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False).strip().replace("</s>", "")
            print(f"DEBUG: Generated text length: {len(text_output)} characters")

            if k == 0:
                print("DEBUG: First question completed, continuing to second question")
                continue
            
            text_output = text_output.replace('[STARTS]', '').replace('[SEG]', '').replace('[ENDS]', '')
            answers.append(text_output)
            print(f"DEBUG: Final text output (first 200 chars): {text_output[:200]}...")

            if True:
                print("DEBUG: Processing and saving results...")
                image_path = data_item['image_path']
                print('DEBUG: image_path', image_path)

                garment_id = image_path.split('/')[-1]
                garment_id = garment_id.split('.')[0]
                print(f"DEBUG: Garment ID: {garment_id}")
                
                json_output = repair_json(text_output, return_objects=True)
                print(f"DEBUG: JSON output keys: {list(json_output.keys()) if isinstance(json_output, dict) else 'Not a dict'}")

                saved_dir = os.path.join(parent_folder, 'vis_new', f'valid_garment_{garment_id}')
                print(f"DEBUG: Saving to directory: {saved_dir}")
                
                if not os.path.exists(saved_dir):
                    os.makedirs(saved_dir)
                    print("DEBUG: Created output directory")
                
                with open(os.path.join(saved_dir, 'output.txt'), 'w') as f:
                    f.write(prompt)
                    f.write('\n')
                    f.write(text_output)
                    f.write('\n')
                    f.write(str(json_output))
                
                print("DEBUG: Saved output.txt")

                output_dir = saved_dir
                all_output_dir.append(output_dir)
                shutil.copy(image_path, os.path.join(output_dir, f'gt_image.png'))
                print("DEBUG: Copied ground truth image")

                all_json_spec_files = run_garmentcode_parser_float50(all_json_spec_files, json_output, float_preds, output_dir)
                print("DEBUG: Processed garment code parser")
        
        print(f"DEBUG: Completed processing image {i+1}/{len_val_dataset}")
        print(f"{'='*60}\n")

    saved_json_Path = os.path.join(parent_folder, 'vis_new', 'all_json_spec_files.json')
    with open(saved_json_Path, 'w') as f:
        json.dump(all_json_spec_files, f)
    
    print("\n" + "="*80)
    print("✓ Inference completed successfully!")
    print(f"✓ Results saved to: {parent_folder}")
    print("="*80 + "\n")

        
if __name__ == "__main__":
    main(sys.argv[1:])
