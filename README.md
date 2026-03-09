# Graviton-Native

**32 GB RAM'da 500B+ parametre. Mimari değişiklikle.**

Graviton-Native, AI modellerini **baştan** verimli mimarilerle eğiten projedir. Post-training quantization yerine, model eğitiminin kendisi düşük bit ve sparse yapıda yapılır — böylece herkes kendi makinesinde büyük modeller çalıştırabilir.

**Technical Report:** [opengraviton.github.io/paper.html](https://opengraviton.github.io/paper.html) | [PAPER.md](PAPER.md)

## Vizyon

| Mevcut Durum | Graviton-Native Hedefi |
|--------------|------------------------|
| 70B model → 140 GB+ RAM | 70B model → **14 GB** (1.58-bit) |
| 500B model → imkansız | 500B MoE → **32 GB** (aktif parametre) |
| Quantization = kalite kaybı | Native training = minimal kayıp |

## Mimari Yaklaşımlar

### 1. BitNet b1.58 — Ternary Weights
- Ağırlıklar **{-1, 0, +1}** ile eğitilir
- Matris çarpımı = sadece toplama/çıkarma (float multiply yok)
- ~10x bellek tasarrufu, ~10x enerji tasarrufu
- Referans: [BitNet b1.58](https://arxiv.org/abs/2402.17764)

### 2. MoE (Mixture of Experts)
- 500B toplam parametre, ~10–20B aktif per token
- Top-K routing: her token için k expert seçilir
- Graviton'da inference desteği

### 3. Sparse / Top-K Activation
- Her katmanda sadece %30 nöron ateşlenir
- %70 hesaplama tasarrufu

## Proje Yapısı

```
graviton-native/
├── graviton_native/
│   ├── models/          # BitNet, MoE mimarileri
│   ├── training/        # Eğitim pipeline
│   └── quantization/    # Eğitim sırasında quantization
├── scripts/             # Eğitim scriptleri
├── configs/             # Model konfigürasyonları
└── README.md
```

## Kurulum

```bash
cd graviton-native
pip install -e ".[train]"
```

## Hızlı Başlangıç

### Hızlı demo (2 adım)

```bash
python scripts/train_bitnet.py --model_size 350m --steps 2
```

### Gerçek veriyle eğitim (WikiText, C4, vb.)

```bash
# WikiText-2 ile 500 adım
python scripts/train_bitnet_full.py --model_size 350m --steps 500

# Özel JSONL veri
python scripts/train_bitnet_full.py --data_path ./data/train.jsonl --model_size 350m
```

### Graviton ile inference

Graviton-Native checkpoint'leri Graviton'da doğrudan çalışır:

```bash
graviton-ui
# BitNet: graviton-native/checkpoints/bitnet-350m
# MoE:    graviton-native/checkpoints/moe-small
# Otomatik algılanır
```

### MoE eğitimi

```bash
python scripts/train_moe.py --model_size small --steps 100
```

## Gereksinimler

- Python 3.9+
- PyTorch 2.0+
- Apple Silicon / NVIDIA GPU (opsiyonel; CPU'da da çalışır)

## Lisans

Apache-2.0 — Graviton ile uyumlu.
