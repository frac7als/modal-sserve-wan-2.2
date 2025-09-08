import subprocess
import os
import modal

# It's good practice to list dependencies in a structured way
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    # Consolidated list of Python dependencies
    .pip_install(
        "gguf",
        "llama-cpp-python",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
    )
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.47"
    )
)

# ## Downloading custom nodes
image = image.run_commands(
    "comfy node install --fast-deps was-node-suite-comfyui@1.0.2",
    "git clone https://github.com/ChenDarYen/ComfyUI-NAG.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-NAG",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone https://github.com/city96/ComfyUI-GGUF.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-GGUF",
    "git clone https://github.com/rgthree/rgthree-comfy.git /root/comfy/ComfyUI/custom_nodes/rgthree-comfy",
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
)

def hf_download():
    from huggingface_hub import hf_hub_download

    wan_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -sf {wan_model} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        shell=True,
        check=True,
    )

    wan_model_low_noise = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -sf {wan_model_low_noise} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        shell=True,
        check=True,
    )

    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -sf {vae_model} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors",
        shell=True,
        check=True,
    )

    t5_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -sf {t5_model} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        shell=True,
        check=True,
    )

    # ======================================================================
    # THIS IS THE CONFIRMED FIX FOR THE MISSING LORA
    # ======================================================================
    # 1. Define the lora directory and ensure it exists
    lora_dir = "/root/comfy/ComfyUI/models/loras"
    os.makedirs(lora_dir, exist_ok=True)

    # 2. Download the lora file from the correct Hugging Face repository
    lora_model_path = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        cache_dir="/cache",
    )
    
    lora_filename = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

    # 3. Create a symbolic link to the lora file in the correct directory
    subprocess.run(
        f"ln -sf {lora_model_path} {os.path.join(lora_dir, lora_filename)}",
        shell=True,
        check=True,
    )
    # ======================================================================


vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .run_function(
        hf_download,
        # persist the HF cache to a Modal Volume so future runs don't re-download models
        volumes={"/cache": vol},
    )
)

app = modal.App(name="example-comfyui", image=image)


@app.function(
    max_containers=1,
    gpu="L40S",
    volumes={"/cache": vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
