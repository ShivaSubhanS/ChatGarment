from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_path_eval: str = field(default=None,
                                metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments:
    """Minimal version for inference only"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    output_dir: str = field(default="./output")
    per_device_eval_batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=1)
    bf16: bool = field(default=True)
    tf16: bool = field(default=True)
    fp16: bool = field(default=False)
    local_rank: int = field(default=0)
    world_size: int = field(default=1)
    ddp_backend: str = field(default="nccl")
    seed: int = field(default=42)
    gradient_checkpointing: bool = field(default=True)  # Required by inference scripts
    fsdp: str = field(default="")  # Fully Sharded Data Parallel
    device: str = field(default="cuda")

    def __post_init__(self):
        """Automatically set cache_dir to pendrive if local model not found"""
        if self.cache_dir is None:
            # Check if LLaVA model exists locally
            local_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b")
            pendrive_model_path = "/media/sss/satti/huggingface_models"

            if os.path.exists(local_model_path):
                self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            elif os.path.exists(pendrive_model_path):
                self.cache_dir = "/media/sss/satti/huggingface_models"
            else:
                # Default to local cache, will download if needed
                self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")


def get_checkpoint_path():
    """Get the checkpoint path, preferring local then pendrive"""
    local_checkpoint = "checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin"
    pendrive_checkpoint = "/media/sss/satti/checkpoints/pytorch_model.bin"

    if os.path.exists(local_checkpoint):
        return local_checkpoint
    elif os.path.exists(pendrive_checkpoint):
        return pendrive_checkpoint
    else:
        raise FileNotFoundError(f"Checkpoint not found locally ({local_checkpoint}) or on pendrive ({pendrive_checkpoint})")


def rank0_print(*args):
    """Simple print function for inference"""
    print(*args)
