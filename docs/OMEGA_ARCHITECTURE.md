# Graviton Omega — Achieving the Impossible

**Goal:** Run an Opus-level (~100B effective) model on 8GB RAM.

## 1. Core Idea

Current LLMs are **dense**: every token uses all parameters. 70B model = 70B × 16 bit = 140 GB.

**Omega idea:** Sparsify parameters so each token uses only ~1%.

```
100B total × 1.58-bit × 1% active = 200 MB RAM
```

## 2. Architecture Components

### 2.1 Ultra-Sparse MoE (k=1)

| Classic MoE | Omega MoE |
|-------------|-----------|
| 8 expert, k=2 → 25% active | 64 expert, k=1 → 1.56% active |
| 70B total, 17B active | 100B total, 1.56B active |

**Formula:** 64 expert × 1.56B = 100B total. Each token goes to 1 expert.

### 2.2 BitNet 1.58-bit

- Weights: {-1, 0, +1}
- 1.56B active × 1.58/8 = **308 MB** per layer

### 2.3 Layer Streaming

- 80 layers, each with 64 experts
- Only the **selected expert** is loaded
- Peak RAM: 1 layer × 1 expert = 308 MB

### 2.4 Activation Cache Minimization

- **Mamba/SSM** hybrid: O(n) context, no KV cache
- Or: **Sparse attention** — only 256 token context

### 2.5 Hierarchical Routing

```
Token → Router(1) → 1 expert (1.56B)
     → Router(2) → "hard?" → Yes: retrieve from memory
     → No: expert continues
```

## 3. Memory Budget (8GB)

| Component | MB |
|-----------|-----|
| 1 expert (1.56B @ 1.58-bit) | 308 |
| Embedding + LM head | 100 |
| Router (64 expert logits) | 1 |
| Activations (batch=1, seq=512) | 8 |
| KV cache (sparse, 256 pos) | 32 |
| Python/PyTorch overhead | 500 |
| **Total** | ~1 GB |

**Remaining 7 GB:** Prefetch, disk buffer, system.

## 4. Disk I/O Optimization

**Problem:** 80 layers × 308 MB = 24 GB read/token. Too slow.

**Solutions:**
1. **Expert locality:** Similar tokens use same expert → cache hit
2. **Layer prefetch:** Load layer i+1 expert while computing layer i
3. **NVMe:** 3 GB/s sequential → 24 GB / 3 = 8 sec/token (still slow)
4. **RAM cache:** 6 GB of 8 GB for expert cache → 20 experts resident = 6 GB

**20 expert cache:** Most frequently used 20 experts in RAM. 80% cache hit = 5 tok/s.

## 5. Training Strategy

- **Stage 1:** Train dense 1.56B BitNet (baseline)
- **Stage 2:** 64 copies, train MoE router (expert specialization)
- **Stage 3:** 100B total, distillation from Opus

## 6. Expected Quality

- 100B total, 1.56B active → **"64 different 1.56B experts"**
- Each expert excels in one domain (code, math, general, etc.)
- If router selects the right expert → much better than 1.56B alone
- **Risk:** Quality drops if router makes mistakes

## 7. Prototype Path

1. **Omega-Micro:** 8 expert × 100M = 800M total, 100M active
2. **Omega-Small:** 16 expert × 500M = 8B total, 500M active  
3. **Omega-100B:** 64 expert × 1.56B = 100B total

---

*"To achieve the impossible, first change the architecture."*
