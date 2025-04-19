import torch
import os
import time
try: # Add try-except for psutil import
    import psutil # For RAM logs
except ImportError:
    print("psutil not found, RAM logging disabled. Run 'pip install psutil' to enable.")
    psutil = None # Set to None if import fails

# Import all required model/tokenizer types from init
from transformers import (
    LlamaForCausalLM, PreTrainedTokenizerFast, LlamaConfig,
    CLIPTextModelWithProjection, CLIPTokenizer, CLIPConfig,
    T5EncoderModel, T5Tokenizer, T5Config,
    BitsAndBytesConfig # Import BitsAndBytesConfig
)
from diffusers.models import AutoencoderKL
from diffusers.schedulers import SchedulerMixin
# Import accelerate utilities AGAIN for T5 offload
from accelerate import init_empty_weights, load_checkpoint_in_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
# Import snapshot_download
from huggingface_hub import snapshot_download

# Assuming these imports point to the correct local modules within the 'hd' directory structure
from . import HiDreamImagePipeline # Import the class itself
# Import Transformer class, we need to load it manually
from . import HiDreamImageTransformer2DModel
# Keep original scheduler imports as they are likely custom classes
from .schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler


MODEL_PREFIX = "azaneko"
# External Llama model
LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
# Other encoders will be loaded from local subfolders

# Model configurations (remains the same)
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler # The class itself
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler # The class itself
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler # The class itself
    }
}


# log_vram function (remains the same, includes detailed breakdown)
def log_vram(msg: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        num_devices = torch.cuda.device_count()
        if num_devices > 0:
            total_mem = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_devices)) / 1024**2
            detailed_mem = [f"GPU{i}: {torch.cuda.memory_allocated(i)/1024**2:.2f}MB" for i in range(num_devices)]
            print(f"{msg} (Total Used: {allocated:.2f} MB across {num_devices} GPUs [{', '.join(detailed_mem)}] - Total Capacity: {total_mem:.2f} MB)\\n")
        else:
            print(f"{msg} (Used {allocated:.2f} MB VRAM - No GPUs detected by PyTorch)\\n")
    else:
        print(f"{msg} (CUDA not available)\\n")

# Helper function for RAM logging
def log_ram(msg: str):
    if psutil: # Check if import succeeded
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"{msg} (Current Process RAM: {mem_info.rss / 1024**2:.2f} MB)")


def load_models(model_type: str):
    overall_start_time = time.time() # Start overall timer
    log_ram(f"Start of load_models for type '{model_type}'")
    print(f"--- Loading model type: {model_type} ---")

    config = MODEL_CONFIGS[model_type]
    model_repo_path = config["path"] # e.g., "azaneko/HiDream-I1-Dev-nf4"

    # --- Check for GPUs ---
    if not torch.cuda.is_available(): raise RuntimeError("CUDA not available.")
    num_gpus = torch.cuda.device_count()
    target_device_main = "cuda:0" # Target device for non-distributed components
    loading_dtype = torch.bfloat16 # Default overall dtype
    if num_gpus < 2:
        print("⚠️ Only one GPU detected.")
        max_memory_llama = None
        offload_folder = None
        max_memory_t5_cpu = None # No specific CPU offload needed for T5 with 1 GPU
    else:
        print(f"✅ Detected {num_gpus} GPUs. Using device_map for Text Encoder 4, placing others on {target_device_main}.")
        # --- Define Max Memory hints ---
        gpu_mem_limit = "14GiB"
        cpu_mem_limit = "28GiB" # Set generous CPU RAM limit
        # Map for Llama (uses GPUs + CPU)
        max_memory_llama = {i: gpu_mem_limit for i in range(num_gpus)}
        max_memory_llama["cpu"] = cpu_mem_limit
        print(f"   Providing max_memory hint to device_map for Text Encoder 4: {max_memory_llama}")
        # Map for T5 (FORCE to CPU as much as possible)
        max_memory_t5_cpu = {"cpu": cpu_mem_limit}
        print(f"   Max memory hint for T5 Encoder (forcing CPU): {max_memory_t5_cpu}")
        # Offload folder (can be shared)
        offload_folder = "./offload_tmp_accelerate"
        os.makedirs(offload_folder, exist_ok=True)
        print(f"   Using offload directory (if needed): {offload_folder}")


    # --- Download HiDream Repo ONCE ---
    start_time = time.time()
    print(f"🔄 Ensuring base HiDream repo is downloaded: {model_repo_path}...")
    local_hidream_repo_path = snapshot_download(repo_id=model_repo_path)
    print(f"   HiDream Repo local path: {local_hidream_repo_path}")
    print(f"   Download/Check time: {time.time() - start_time:.2f}s")
    try: print(f"   Contents: {os.listdir(local_hidream_repo_path)}")
    except Exception as e: print(f"   Warning: Could not list directory contents: {e}")
    print("")


    # --- Load ALL Tokenizers (CPU) ---
    # ... (Keep same as previous version) ...
    start_time = time.time()
    print("🔄 Loading ALL Tokenizers...")
    log_ram("Before tokenizers load")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(local_hidream_repo_path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(local_hidream_repo_path, subfolder="tokenizer_2")
        tokenizer_3 = T5Tokenizer.from_pretrained(local_hidream_repo_path, subfolder="tokenizer_3")
    except Exception as e: print(f"❌ FAILED to load one of the CLIP/T5 tokenizers from subfolders: {e}"); raise e
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME)
    log_vram("✅ All tokenizers loaded!")
    log_ram("After all tokenizers load")
    print(f"   All tokenizers load time: {time.time() - start_time:.2f}s\\n")


    # --- Load Text Encoder 4 (Llama) using device_map="auto" ---
    # ... (Keep same as previous version) ...
    start_time = time.time()
    print("🔄 Loading Text Encoder 4 (Llama) with device_map='auto'...")
    log_ram("Before text encoder 4 load")
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME, output_hidden_states=True, output_attentions=True, return_dict_in_generate=True,
        torch_dtype=loading_dtype, device_map="auto" if num_gpus >= 2 else None,
        max_memory=max_memory_llama if num_gpus >= 2 else None, offload_folder=offload_folder if num_gpus >= 2 else None,
    )
    log_vram("✅ Text encoder 4 loaded (using device_map)!")
    log_ram("After text encoder 4 load")
    print(f"   Text encoder 4 load time: {time.time() - start_time:.2f}s\\n")


    # --- Load Text Encoder 1 & 2 (CLIP L/G) onto target_device_main ---
    # ... (Keep same as previous version) ...
    print(f"🔄 Loading CLIP Text Encoders onto {target_device_main} from LOCAL subfolders...")
    start_time = time.time()
    log_ram("Before CLIP encoders load")
    try:
        text_encoder = CLIPTextModelWithProjection.from_pretrained(local_hidream_repo_path, subfolder="text_encoder", torch_dtype=loading_dtype, use_safetensors=True).to(target_device_main)
        log_vram("After text_encoder (CLIP-L) load")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(local_hidream_repo_path, subfolder="text_encoder_2", torch_dtype=loading_dtype, use_safetensors=True).to(target_device_main)
        log_vram("✅ CLIP text encoders loaded!")
        log_ram("After CLIP encoders load")
        print(f"   CLIP encoders load time: {time.time() - start_time:.2f}s\\n")
    except Exception as e: print(f"❌ FAILED to load CLIP text encoders from subfolders: {e}"); raise e


    # --- Load Text Encoder 3 (T5-XXL) using Accelerate -> CPU Offload ---
    # ... (Keep same as previous version) ...
    start_time = time.time()
    print(f"🔄 Loading Text Encoder 3 (T5) via Accelerate with forced CPU offload...")
    log_ram("Before text encoder 3 load")
    t5_subfolder_path = os.path.join(local_hidream_repo_path, "text_encoder_3")
    print(f"   T5 path: {t5_subfolder_path}")
    try:
        with init_empty_weights():
            t5_config = T5Config.from_pretrained(t5_subfolder_path)
            text_encoder_3 = T5EncoderModel(t5_config)
        if num_gpus >= 2:
            device_map_t5 = infer_auto_device_map(text_encoder_3, max_memory=max_memory_t5_cpu, no_split_module_classes=["T5Block"])
            print(f"   T5 device map (forced CPU): {device_map_t5}")
        else: device_map_t5 = None; print(f"   Loading T5 directly to {target_device_main}")
        load_checkpoint_in_model(
             text_encoder_3, checkpoint=t5_subfolder_path, device_map=device_map_t5,
             offload_folder=offload_folder, dtype=loading_dtype, offload_state_dict=True
        )
        if num_gpus < 2: text_encoder_3.to(target_device_main)
        log_vram("✅ Text encoder 3 loaded!")
    except Exception as e:
        print(f"❌ FAILED to load T5 Encoder: {e}")
        try: # Fallback to CPU
            print("   Retrying T5 load directly to CPU...")
            text_encoder_3 = T5EncoderModel.from_pretrained(t5_subfolder_path, torch_dtype=loading_dtype, use_safetensors=True, low_cpu_mem_usage=True).to('cpu')
            log_vram("✅ Text encoder 3 loaded (Direct to CPU Fallback)!")
        except Exception as e2: print(f"❌ FAILED fallback T5 load to CPU: {e2}"); raise e
    log_ram("After text encoder 3 load")
    print(f"   Text encoder 3 load time: {time.time() - start_time:.2f}s\\n")


    # --- Load VAE onto SINGLE GPU ---
    # ... (Keep same as previous version) ...
    start_time = time.time()
    print(f"🔄 Loading VAE onto {target_device_main} from LOCAL subfolder...")
    log_ram("Before VAE load")
    vae = AutoencoderKL.from_pretrained(
         local_hidream_repo_path, subfolder="vae", torch_dtype=loading_dtype,
         low_cpu_mem_usage=True, use_safetensors=True
    ).to(target_device_main)
    log_vram(f"✅ VAE loaded onto {target_device_main}!")
    log_ram("After VAE load")
    print(f"   VAE load time: {time.time() - start_time:.2f}s\\n")


    # --- Load TRANSFORMER onto SINGLE GPU with BitsAndBytes --- ### MODIFIED BnB Config & Removed .to() ###
    start_time = time.time()
    print(f"🔄 Loading Transformer onto {target_device_main} from LOCAL subfolder with explicit BnB config...")
    log_ram("Before transformer load")
    # Create BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # <<< USING float16 based on original config
        bnb_4bit_use_double_quant=False,
    )
    print(f"   Using BitsAndBytesConfig: {bnb_config}")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        local_hidream_repo_path,
        subfolder="transformer",
        quantization_config=bnb_config, # Pass explicit config
        low_cpu_mem_usage=True, # Try to minimize RAM usage before loading to GPU
    )
    # REMOVED explicit .to(target_device_main) here
    # Check device after loading and move if necessary
    transformer_device = next(transformer.parameters()).device
    print(f"   Transformer loaded initially to device: {transformer_device}")
    if transformer_device != torch.device(target_device_main):
        print(f"   Moving transformer from {transformer_device} to {target_device_main}...")
        transformer.to(target_device_main)

    log_vram(f"✅ Transformer loaded onto {target_device_main}!")
    log_ram("After transformer load")
    print(f"   Transformer load time: {time.time() - start_time:.2f}s\\n")


    # --- Instantiate Scheduler ---
    # ... (Keep same as previous version) ...
    start_time = time.time()
    print("🔄 Initializing scheduler...")
    scheduler_class = config["scheduler"]
    scheduler = None
    try:
        scheduler = scheduler_class.from_pretrained(local_hidream_repo_path, subfolder="scheduler")
        scheduler.config.shift = config["shift"]
        scheduler.config.use_dynamic_shifting = False
        print("   Loaded scheduler from config.")
    except Exception as e:
         print(f" W Could not load scheduler from config ({e}), instantiating directly.")
         scheduler = scheduler_class(num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    log_ram("After scheduler init")
    print(f"   Scheduler init time: {time.time() - start_time:.2f}s\\n")


    # --- Instantiate Pipeline MANUALLY ---
    # ... (Keep same as previous version) ...
    start_time = time.time()
    print("🔄 Instantiating HiDreamImagePipeline manually with ALL components...")
    log_ram("Before pipeline instantiation")
    pipe = HiDreamImagePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
        text_encoder_3=text_encoder_3, tokenizer_3=tokenizer_3,
        text_encoder_4=text_encoder_4, tokenizer_4=tokenizer_4,
        scheduler=scheduler,
    )
    print("   Assigning transformer model to pipe.transformer...")
    pipe.transformer = transformer
    if hasattr(pipe, 'transformer') and pipe.transformer is not None: print("   Successfully assigned transformer to pipe.transformer")
    else: print("⚠️ WARNING: Failed to assign transformer to pipe.transformer!")
    log_vram("✅ Pipeline manually instantiated & transformer assigned!")
    log_ram("After pipeline instantiation")
    print(f"   Pipeline instantiation time: {time.time() - start_time:.2f}s\\n")

    print(f"✅ Load strategy: TextEnc4 distributed (if multi-GPU), TextEnc3 CPU-offloaded (if multi-GPU), others on {target_device_main}.")
    print(f"--- Total load_models time: {time.time() - overall_start_time:.2f}s ---")
    return pipe, config


@torch.inference_mode()
def generate_image(pipe: HiDreamImagePipeline, model_type: str, prompt: str, resolution: tuple[int, int], seed: int):
    # (generate_image function remains the same - using default generator)
    # ... (definition as before) ...
    overall_start_time = time.time()
    log_ram("Start of generate_image")
    config = MODEL_CONFIGS[model_type]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    width, height = resolution
    if seed == -1: seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator().manual_seed(seed)
    print(f"⏳ Starting image generation (Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed})...")
    gen_start_time = time.time()
    images = pipe(
        prompt, height=height, width=width, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
        num_images_per_prompt=1, generator=generator,
    ).images
    print(f"   Pipeline call time: {time.time() - gen_start_time:.2f}s")
    print("✅ Image generation complete.")
    log_ram("After image generation")
    if images is None or len(images) == 0: raise RuntimeError("Image generation failed.")
    print(f"--- Total generate_image time: {time.time() - overall_start_time:.2f}s ---")
    return images[0], seed
