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
    """Kaggle-specific configuration for GPU inference"""
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
    lora_r: int = 128  # Updated to match checkpoint
    lora_alpha: int = 256  # Updated to match checkpoint
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    
    # Required attributes for inference
    local_rank: int = field(default=-1)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mm_use_im_start_end: bool = field(default=False)
    use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    fsdp: str = field(default="")  # Fully Sharded Data Parallel
    device: str = field(default="cuda")  # Changed to cuda for Kaggle

    def __post_init__(self):
        """Kaggle-specific path configuration"""
        if self.cache_dir is None:
            # Kaggle dataset paths
            kaggle_huggingface_path = "/kaggle/input/shivasubhans/llava_huggingface"
            
            if os.path.exists(kaggle_huggingface_path):
                self.cache_dir = kaggle_huggingface_path
                print(f"✓ Using Kaggle Hugging Face cache: {self.cache_dir}")
            else:
                # Fallback to default cache
                self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                print(f"⚠ Kaggle path not found, using default: {self.cache_dir}")


def get_checkpoint_path():
    """Get checkpoint path from Kaggle dataset"""
    kaggle_checkpoint = "/kaggle/input/shivasubhans/llava_finetuned/pytorch_model.bin"
    
    if os.path.exists(kaggle_checkpoint):
        print(f"✓ Using Kaggle checkpoint: {kaggle_checkpoint}")
        return kaggle_checkpoint
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {kaggle_checkpoint}")


def rank0_print(*args):
    """Simple print function for inference"""
    print(*args)
