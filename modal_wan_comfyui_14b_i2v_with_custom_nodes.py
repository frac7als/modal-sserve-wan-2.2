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
        "wget",  # Add wget to download the sample video
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
# Now, clone the nodes into the image that already has their dependencies
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

# ========================================================================
# FIX 2: Download a sample video into the ComfyUI input directory
# This makes a default video available for the VHS_LoadVideo node.
# ========================================================================
image = image.run_commands(
    "wget -P /root/comfy/ComfyUI/input/ https://github.com/modal-labs/modal-examples/raw/main/comfyui/comfyui_input.mp4",
)

def hf_download():
    from huggingface_hub import hf_hub_download
    import os

    # ========================================================================
    # FIX 1: Download GGUF models and place them in the correct directory
    # ========================================================================
    GGUF_MODEL_DIR = "/root/comfy/ComfyUI/models/gguf"
    os.makedirs(GGUF_MODEL_DIR, exist_ok=True)

    # Download the GGUF version of the high-noise model
    wan_gguf_model = hf_hub_download(
        repo_id="bullerwins/Wan2.2-I2V-A14B-GGUF",
        filename="wan2.2_i2v_low_noise_14B_Q8_0.gguf",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_gguf_model} {GGUF_MODEL_DIR}/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
        shell=True,
        check=True,
    )

    # Download the GGUF version of the low-noise model
    wan_gguf_low_noise = hf_hub_download(
        repo_id="bullerwins/Wan2.2-I2V-A14B-GGUF",
        filename="wan2.2_i2v_low_noise_14B_Q8_0.gguf",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_gguf_low_noise} {GGUF_MODEL_DIR}/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
        shell=True,
        check=True,
    )

    # We still need the VAE and Text Encoder, which are not GGUF files
    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {vae_model} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors",
        shell=True,
        check=True,
    )

    t5_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {t5_model} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        shell=True,
        check=True,
    )

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

## Running ComfyUI interactively
app = modal.App(name="example-comfyui", image=image)

@app.function(
    max_containers=1,  # limit interactive session to 1 container
    gpu="L40S",  # good starter GPU for inference
    volumes={"/cache": vol},  # mounts our cached models
)
@modal.concurrent(
    max_inputs=10
)  # required for UI startup process which runs several API calls concurrently
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
