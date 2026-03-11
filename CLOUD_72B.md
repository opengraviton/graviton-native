# 72B Training — Cloud Setup

72B model training requires 8x A100 80GB. Run on cloud:

---

## Option 1: RunPod (Recommended)

1. **Create account:** [runpod.io](https://runpod.io)
2. **Deploy Pod:** 
   - GPU: 8x A100 80GB SXM
   - Image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
   - Volume: 100GB+ (for checkpoints)
3. **Connect:** SSH or Web Terminal
4. **Run:**

```bash
# One-time: clone your repo (or upload)
cd /workspace
git clone https://github.com/opengraviton/graviton-native.git
cd graviton-native

# HF token (request access at https://huggingface.co/datasets/bigcode/the-stack)
export HF_TOKEN=hf_xxxxxxxxxxxx

# Start training
bash scripts/run_72b_cloud.sh
```

**Cost:** ~$12–15/hr for 8x A100 80GB. 100K steps ≈ 3–4 weeks ≈ $6K–12K.

---

## Option 2: Vast.ai

1. **Create account:** [vast.ai](https://vast.ai)
2. **Rent instance:** Search for 8x A100 80GB
3. **SSH in**, then:

```bash
cd /workspace
git clone https://github.com/opengraviton/graviton-native.git
cd graviton-native
export HF_TOKEN=hf_xxx
bash scripts/run_72b_cloud.sh
```

**Cost:** Often cheaper (~$8–10/hr) but can be interrupted.

---

## Option 3: Lambda Labs

1. **Create account:** [lambdalabs.com](https://lambdalabs.com)
2. **Launch:** 8x A100 80GB instance
3. **SSH in**, same commands as above.

**Cost:** ~$16/hr, stable, no interruptions.

---

## Option 4: Your Own Repo (Local Upload)

If graviton-native is on your machine:

```bash
# From your Mac
cd /path/to/your/ai
scp -r graviton-native user@cloud-instance:/workspace/
```

Then on cloud:

```bash
cd /workspace/graviton-native
export HF_TOKEN=hf_xxx
bash scripts/run_72b_cloud.sh
```

---

## Quick Start (Copy-Paste)

On any 8x A100 cloud instance:

```bash
cd /workspace
git clone https://github.com/opengraviton/graviton-native.git
cd graviton-native
export HF_TOKEN=hf_YOUR_TOKEN
pip install -e ".[train]" deepspeed -q
graviton-train run --num_gpus 8 --model_size 72b --dataset the-stack --steps 100000 --batch_size 1 --grad_accum 64 --seq_len 2048 --output_dir /workspace/checkpoints --save_every 1000
```

---

## Checkpoints

After training, download:

```bash
# From cloud to your Mac
scp -r user@cloud:/workspace/checkpoints/bitnet-72b ./graviton-native/checkpoints/
```

Then run in Graviton:

```bash
graviton-ui
# Model: ./checkpoints/bitnet-72b/final
```
