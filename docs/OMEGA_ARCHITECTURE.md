# Graviton Omega — İmkansızı Başarmak

**Hedef:** Opus seviyesi (~100B efektif) modeli 8GB RAM'de çalıştırmak.

## 1. Temel Fikir

Mevcut LLM'ler **dense**: Her token için tüm parametreler kullanılır. 70B model = 70B × 16 bit = 140 GB.

**Omega fikri:** Parametreleri o kadar seyrelt ki, her token sadece %1'ini kullansın.

```
100B total × 1.58-bit × 1% active = 200 MB RAM
```

## 2. Mimari Bileşenleri

### 2.1 Ultra-Sparse MoE (k=1)

| Klasik MoE | Omega MoE |
|-----------|-----------|
| 8 expert, k=2 → 25% active | 64 expert, k=1 → 1.56% active |
| 70B total, 17B active | 100B total, 1.56B active |

**Formül:** 64 expert × 1.56B = 100B total. Her token 1 expert'e gider.

### 2.2 BitNet 1.58-bit

- Ağırlıklar: {-1, 0, +1}
- 1.56B active × 1.58/8 = **308 MB** per layer

### 2.3 Layer Streaming

- 80 layer, her biri 64 expert
- Sadece **seçilen expert** yüklenir
- Peak RAM: 1 layer × 1 expert = 308 MB

### 2.4 Activation Cache Minimizasyonu

- **Mamba/SSM** hybrid: O(n) context, KV cache yok
- Veya: **Sparse attention** — sadece 256 token context

### 2.5 Hierarchical Routing

```
Token → Router(1) → 1 expert (1.56B)
     → Router(2) → "zor mu?" → Evet: retrieve from memory
     → Hayır: expert devam
```

## 3. Bellek Bütçesi (8GB)

| Bileşen | MB |
|---------|-----|
| 1 expert (1.56B @ 1.58-bit) | 308 |
| Embedding + LM head | 100 |
| Router (64 expert logits) | 1 |
| Activations (batch=1, seq=512) | 8 |
| KV cache (sparse, 256 pos) | 32 |
| Python/PyTorch overhead | 500 |
| **Toplam** | ~1 GB |

**Kalan 7 GB:** Prefetch, disk buffer, sistem.

## 4. Disk I/O Optimizasyonu

**Sorun:** 80 layer × 308 MB = 24 GB read/token. Çok yavaş.

**Çözümler:**
1. **Expert locality:** Benzer tokenlar aynı expert → cache hit
2. **Layer prefetch:** Layer i compute edilirken layer i+1 expert yükle
3. **NVMe:** 3 GB/s sequential → 24 GB / 3 = 8 saniye/token (hala yavaş)
4. **RAM cache:** 8GB'ın 6GB'ı expert cache → 20 expert resident = 6 GB

**20 expert cache:** En sık kullanılan 20 expert RAM'de. %80 cache hit = 5 token/s.

## 5. Eğitim Stratejisi

- **Aşama 1:** Dense 1.56B BitNet eğit (baseline)
- **Aşama 2:** 64 kopya, MoE router eğit (expert specialization)
- **Aşama 3:** 100B total, distillation from Opus

## 6. Beklenen Kalite

- 100B total, 1.56B active → **"64 farklı 1.56B uzman"**
- Her uzman bir domain'de iyi (kod, math, genel, vb.)
- Router doğru uzmanı seçerse → 1.56B'den çok daha iyi
- **Risk:** Router hata yaparsa kalite düşer

## 7. Prototip Yolu

1. **Omega-Micro:** 8 expert × 100M = 800M total, 100M active
2. **Omega-Small:** 16 expert × 500M = 8B total, 500M active  
3. **Omega-100B:** 64 expert × 1.56B = 100B total

---

*"İmkansızı başarmak için önce mimariyi değiştirmek gerekir."*
