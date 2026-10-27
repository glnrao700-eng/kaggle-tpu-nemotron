import os
import sys
import time
import subprocess
import urllib.request
from huggingface_hub import snapshot_download

# Configuration
# Note: Replace MODEL_ID with the exact HuggingFace repo ID for the 4-bit quantized weights
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/Nemotron-3.5-Super-VL-120B-INT4")
LOCAL_MODEL_DIR = "/tmp/huggingface/nemotron_model"
INFERENCE_PORT = 8000
WEBUI_PORT = 8080

def wait_for_port(port, name, timeout=900):
    """Polls localhost:port until the service responds."""
    print(f"[*] Waiting for {name} on port {port} to become ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            print(f"[+] {name} is LIVE on port {port}!")
            return True
        except Exception:
            # Open WebUI doesn't use /health on all versions, check root if needed
            try:
                if port == WEBUI_PORT:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
                    print(f"[+] {name} is LIVE on port {port}!")
                    return True
            except Exception:
                pass
            time.sleep(5)
    print(f"[-] Timeout waiting for {name} on port {port}.")
    return False

def main():
    # Enable fast Rust-based Hugging Face downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print(f"[*] Downloading 4-bit quantized model weights: {MODEL_ID}")
    print("    (Saving to /tmp/huggingface to fit within Kaggle disk limits...)")
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_MODEL_DIR,
            ignore_patterns=["*.pt", "*.bin"] # Prefer safetensors
        )
    except Exception as e:
        print(f"[!] Note: Could not auto-download {MODEL_ID} ({e}).")
        print("    Make sure HF_TOKEN is set if the repo is gated, or verify the exact HF repo ID.")

    # 1. Start the TPU Inference Server (SGLang / vLLM OpenAI-compatible API)
    # Using all 8 TPU v5e cores (tensor-parallel-size=8 -> 128GB total HBM)
    print("[*] Starting TPU Inference Server across 8 TPU v5e cores (128GB Memory)...")
    inference_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", LOCAL_MODEL_DIR,
        "--device", "tpu",
        "--tensor-parallel-size", "8",
        "--quantization", "awq", # or "int4" / "compressed-tensors" depending on weight format
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--port", str(INFERENCE_PORT),
        "--trust-remote-code"
    ]
    inference_proc = subprocess.Popen(
        inference_cmd,
        stdout=open("/tmp/inference.log", "w"),
        stderr=subprocess.STDOUT
    )

    if not wait_for_port(INFERENCE_PORT, "TPU Inference Server", timeout=1200):
        print("[-] Inference server failed to start. Printing last 40 lines of /tmp/inference.log:")
        os.system("tail -n 40 /tmp/inference.log")
        sys.exit(1)

    # 2. Start Open WebUI pre-connected to the local TPU Inference Server
    print("[*] Starting Open WebUI server...")
    webui_env = os.environ.copy()
    webui_env["OPENAI_API_BASE_URL"] = f"http://127.0.0.1:{INFERENCE_PORT}/v1"
    webui_env["OPENAI_API_KEY"] = "sk-local-tpu-key"
    webui_env["WEBUI_AUTH"] = "False" # Skip login screen for quick personal Kaggle use

    webui_cmd = ["open-webui", "serve", "--port", str(WEBUI_PORT)]
    webui_proc = subprocess.Popen(
        webui_cmd,
        env=webui_env,
        stdout=open("/tmp/webui.log", "w"),
        stderr=subprocess.STDOUT
    )

    wait_for_port(WEBUI_PORT, "Open WebUI", timeout=300)

    # 3. Start Cloudflare Tunnel so you get a clickable public URL in Kaggle
    print("\n====================================================")
    print(" [5/5] Exposing Open WebUI via Public URL...        ")
    print(" Click the '*.trycloudflare.com' link below to chat!")
    print("====================================================\n")
    
    tunnel_cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{WEBUI_PORT}"]
    tunnel_proc = subprocess.Popen(tunnel_cmd)

    # Keep the Kaggle cell alive indefinitely while running
    try:
        tunnel_proc.wait()
    except KeyboardInterrupt:
        print("Shutting down servers...")
        tunnel_proc.terminate()
        webui_proc.terminate()
        inference_proc.terminate()

if __name__ == "__main__":
    main()