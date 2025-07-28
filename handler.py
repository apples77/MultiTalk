import os
import subprocess
from huggingface_hub import snapshot_download

class EndpointHandler:
    def __init__(self, path="."):
        # This init method is called when the endpoint is initialized.
        # It's where you'll download models and set up the environment.
        
        # 1. Install dependencies
        # Using -q for quiet to keep logs clean
        subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"])
        subprocess.run(["pip", "install", "-q", "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
        subprocess.run(["pip", "install", "-q", "-U", "xformers==0.0.28", "--index-url", "https://download.pytorch.org/whl/cu121"])
        subprocess.run(["pip", "install", "-q", "flash_attn==2.7.4.post1"])
        subprocess.run(["conda", "install", "-c", "conda-forge", "librosa", "-y"])
        subprocess.run(["conda", "install", "-c", "conda-forge", "ffmpeg", "-y"])

        # 2. Download all the required model weights
        # This is crucial. All models are downloaded into a local 'weights' directory.
        self.weights_dir = os.path.join(path, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Note: snapshot_download is efficient and uses caching.
        snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir=os.path.join(self.weights_dir, "Wan2.1-I2V-14B-480P"), local_dir_use_symlinks=False)
        snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir=os.path.join(self.weights_dir, "chinese-wav2vec2-base"), local_dir_use_symlinks=False)
        snapshot_download("hexgrad/Kokoro-82M", local_dir=os.path.join(self.weights_dir, "Kokoro-82M"), local_dir_use_symlinks=False)
        snapshot_download("MeiGen-AI/MeiGen-MultiTalk", local_dir=os.path.join(self.weights_dir, "MeiGen-MultiTalk"), local_dir_use_symlinks=False)
        
        # The original repo has a step to link/copy files. We do that here.
        multitalk_path = os.path.join(self.weights_dir, "MeiGen-MultiTalk")
        wan_path = os.path.join(self.weights_dir, "Wan2.1-I2V-14B-480P")
        
        os.rename(os.path.join(wan_path, "diffusion_pytorch_model.safetensors.index.json"), os.path.join(wan_path, "diffusion_pytorch_model.safetensors.index.json_old"))
        os.link(os.path.join(multitalk_path, "diffusion_pytorch_model.safetensors.index.json"), os.path.join(wan_path, "diffusion_pytorch_model.safetensors.index.json"))
        os.link(os.path.join(multitalk_path, "multitalk.safetensors"), os.path.join(wan_path, "multitalk.safetensors"))

        # The model is now set up. The __call__ method will handle inference.
        print("Environment and models are set up.")


    def __call__(self, data):
        # This method is called for every incoming request.
        # 'data' is the payload of the request, which you'll define.
        
        # 1. Parse inputs from the request payload
        # Example: Expecting a JSON with 'input_json_path' and 'output_filename'
        # You'll need to handle file inputs (like audio and images) by decoding them, e.g., from base64.
        input_json_path = data.pop("input_json_path", "examples/single_example_1.json")
        save_file = data.pop("save_file", "output/result")
        sample_steps = data.pop("sample_steps", 40)
        
        # Construct the command to run the original inference script
        cmd = [
            "python", "generate_multitalk.py",
            "--ckpt_dir", os.path.join(self.weights_dir, "Wan2.1-I2V-14B-480P"),
            "--wav2vec_dir", os.path.join(self.weights_dir, "chinese-wav2vec2-base"),
            "--input_json", input_json_path,
            "--sample_steps", str(sample_steps),
            "--mode", "streaming",
            "--use_teacache",
            "--save_file", save_file
        ]
        
        # 2. Run the inference script
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # If the script fails, return the error
            return {"error": result.stderr}
        
        # 3. Process the output
        # The script saves the video to a file. You need to read this file
        # and return it, likely as a base64-encoded string.
        # The output path will be something like f"results/{save_file}_0.mp4"
        # You would read this file, encode it, and return it as JSON.
        # This part is complex and requires careful handling of file paths.
        
        # For now, we just return the script's output log.
        return {"status": "success", "log": result.stdout}
