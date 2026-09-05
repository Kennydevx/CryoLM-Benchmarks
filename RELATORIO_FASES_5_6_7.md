# 🧊 RELATÓRIO METROLÓGICO CONSOLIDADO: FASES 5, 6 E 7
> **Data:** 2026-09-03 | **Hardware:** NVIDIA GeForce RTX 3060 (12GB VRAM)

## 1. FASE 5 — SEPARAÇÃO DE GPU DE MEMÓRIA TOTAL
$$M_{\text{total}} = M_{\text{GPU}} + M_{\text{RAM}} + M_{\text{NVMe}}$$

| Modelo | L (tokens) | VRAM Ativa | VRAM Resv | RAM Proc | RAM Sist | NVMe Escrito | I/O (MB/s) | I/O ops/s | Latência Retr | Índice | $M_{\text{total}}$ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cryo | 512 | 70.6 MB | 82.0 MB | 696 MB | 1252 MB | 0.00 MB | 0.00 | 88 | 30.58 ms | 8.00 MB | **767.0 MB** |
| cryo | 1024 | 82.2 MB | 106.0 MB | 697 MB | 1240 MB | 0.00 MB | 0.00 | 139 | 11.46 ms | 8.00 MB | **779.2 MB** |
| cryo | 2048 | 105.3 MB | 138.0 MB | 697 MB | 1244 MB | 0.00 MB | 0.00 | 117 | 13.54 ms | 8.00 MB | **802.2 MB** |
| cryo | 4096 | 151.4 MB | 210.0 MB | 696 MB | 1245 MB | 0.00 MB | 0.00 | 85 | 12.54 ms | 8.00 MB | **847.8 MB** |
| cryo | 8192 | 245.3 MB | 358.0 MB | 697 MB | 1255 MB | 0.00 MB | 0.00 | 56 | 12.70 ms | 8.12 MB | **942.8 MB** |
| cryo | 16384 | 450.6 MB | 638.0 MB | 715 MB | 1285 MB | 0.00 MB | 0.00 | 32 | 16.79 ms | 16.12 MB | **1166.0 MB** |
| cryo | 32768 | 852.1 MB | 1228.0 MB | 738 MB | 1320 MB | 0.00 MB | 0.00 | 18 | 14.33 ms | 32.12 MB | **1590.6 MB** |
| cryo | 65536 | 1661.8 MB | 2418.0 MB | 804 MB | 1373 MB | 0.00 MB | 0.00 | 10 | 21.12 ms | 64.12 MB | **2465.8 MB** |
| dense | 512 | 68.7 MB | 88.0 MB | 658 MB | 1237 MB | 0.00 MB | 0.00 | 167 | - | - | **727.1 MB** |
| dense | 1024 | 131.2 MB | 148.0 MB | 659 MB | 1253 MB | 0.00 MB | 0.00 | 112 | - | - | **790.1 MB** |
| dense | 2048 | 369.5 MB | 384.0 MB | 659 MB | 1244 MB | 0.00 MB | 0.00 | 44 | - | - | **1028.1 MB** |
| dense | 4096 | 1302.8 MB | 1476.0 MB | 659 MB | 1231 MB | 0.00 MB | 0.00 | 13 | - | - | **1961.7 MB** |
| dense | 8192 | 4992.7 MB | 5152.0 MB | 668 MB | 1239 MB | 0.00 MB | 0.00 | 4 | - | - | **5661.2 MB** |
| dense | 16384 | 19688.2 MB | 20250.0 MB | 659 MB | 1207 MB | 0.00 MB | 0.00 | 0 | - | - | **20347.4 MB** |
| dense | 32768 | **OOM** | - | - | - | - | - | - | - | - | **CRASH** |
| dense | 65536 | **OOM** | - | - | - | - | - | - | - | - | **CRASH** |

---

## 2. FASE 6 — ESCALABILIDADE DE CONTEXTO EXTREMO & NEEDLE RECALL

| Modelo | Contexto L | VRAM Ativa | RAM Proc | NVMe Escrito | Throughput | Latência Total | Needle Recuperado | Acurácia Needle | Status |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| cryo | 16384 | 450.6 MB | 714 MB | 0.00 MB | 146514 tok/s | 111.8 ms | 1/1 | **100.0%** | ✅ |
| cryo | 32768 | 852.3 MB | 740 MB | 0.00 MB | 153567 tok/s | 213.4 ms | 1/1 | **100.0%** | ✅ |
| cryo | 65536 | 1662.5 MB | 802 MB | 0.00 MB | 158273 tok/s | 414.1 ms | 1/1 | **100.0%** | ✅ |
| cryo | 131072 | 3283.0 MB | 932 MB | 0.00 MB | 160064 tok/s | 818.9 ms | 1/1 | **100.0%** | ✅ |
| cryo | 262144 | 6524.7 MB | 1175 MB | 0.00 MB | 171844 tok/s | 1525.5 ms | 1/1 | **100.0%** | ✅ |
| dense | 16384 | 19688.2 MB | 659 MB | 0.00 MB | 693 tok/s | 23656.9 ms | 0/14 | **0.0%** | ✅ |
| dense | 32768 | - | - | - | - | - | 0/0 | 0.0% | ❌ OOM |
| dense | 65536 | - | - | - | - | - | 0/0 | 0.0% | ❌ OOM |
| dense | 131072 | - | - | - | - | - | 0/0 | 0.0% | ❌ OOM |
| dense | 262144 | - | - | - | - | - | 0/0 | 0.0% | ❌ OOM |

---

## 3. FASE 7 — PREFILL vs DECODE (TIMING PRECISO VIA CUDA EVENTS)

### A. Métricas de Prefill (Prompt Processing)

| Modelo | Prompt L | Tempo Prefill | Throughput Prefill | VRAM Prefill | RAM Prefill |
|:---|---:|---:|---:|---:|---:|
| cryo | 512 | 10.6 ms | **48137 tok/s** | 54.4 MB | 676 MB |
| cryo | 1024 | 13.9 ms | **73916 tok/s** | 67.1 MB | 677 MB |
| cryo | 2048 | 18.8 ms | **109225 tok/s** | 93.0 MB | 679 MB |
| cryo | 4096 | 29.7 ms | **137850 tok/s** | 143.0 MB | 685 MB |
| cryo | 8192 | 50.0 ms | **163985 tok/s** | 246.5 MB | 696 MB |
| cryo | 16384 | 90.8 ms | **180371 tok/s** | 450.6 MB | 714 MB |
| cryo | 32768 | 186.4 ms | **175764 tok/s** | 852.3 MB | 737 MB |
| cryo | 65536 | 344.9 ms | **190020 tok/s** | 1662.6 MB | 800 MB |
| dense | 512 | 11.8 ms | **43558 tok/s** | 59.3 MB | 669 MB |
| dense | 1024 | 13.3 ms | **77137 tok/s** | 123.2 MB | 668 MB |
| dense | 2048 | 37.2 ms | **55127 tok/s** | 363.7 MB | 669 MB |
| dense | 4096 | 138.1 ms | **29651 tok/s** | 1298.9 MB | 669 MB |
| dense | 8192 | **OOM** | - | - | - |
| dense | 16384 | **OOM** | - | - | - |
| dense | 32768 | **OOM** | - | - | - |
| dense | 65536 | **OOM** | - | - | - |

### B. Métricas de Decode (Geração Autoregressiva de 64 Tokens)

| Modelo | Prompt L | TTFT (1º Token) | Média ITL | p50 | p95 | p99 | Throughput Decode |
|:---|---:|---:|---:|---:|---:|---:|---:|
| cryo | 512 | 93.44 ms | 3.48 ms | 3.46 ms | 3.70 ms | 3.74 ms | **205 tok/s** |
| cryo | 1024 | 98.82 ms | 3.62 ms | 3.60 ms | 3.75 ms | 3.88 ms | **196 tok/s** |
| cryo | 2048 | 97.41 ms | 3.70 ms | 3.62 ms | 3.95 ms | 5.30 ms | **194 tok/s** |
| cryo | 4096 | 100.86 ms | 4.29 ms | 3.95 ms | 6.67 ms | 8.53 ms | **172 tok/s** |
| cryo | 8192 | 96.13 ms | 3.55 ms | 3.53 ms | 3.68 ms | 3.78 ms | **200 tok/s** |
| cryo | 16384 | 92.56 ms | 4.28 ms | 4.13 ms | 5.41 ms | 8.06 ms | **177 tok/s** |
| cryo | 32768 | 107.05 ms | 3.58 ms | 3.49 ms | 3.74 ms | 5.39 ms | **193 tok/s** |
| cryo | 65536 | 96.40 ms | 3.63 ms | 3.53 ms | 3.81 ms | 5.62 ms | **197 tok/s** |
| dense | 512 | 4.54 ms | 4.80 ms | 4.51 ms | 5.99 ms | 8.29 ms | **209 tok/s** |
| dense | 1024 | 13.49 ms | 10.02 ms | 9.88 ms | 10.93 ms | 12.64 ms | **99 tok/s** |
| dense | 2048 | 38.86 ms | 31.21 ms | 30.53 ms | 39.07 ms | 42.82 ms | **32 tok/s** |
| dense | 4096 | 128.16 ms | 105.56 ms | 101.27 ms | 137.85 ms | 145.91 ms | **9 tok/s** |
| dense | 8192 | **OOM** | - | - | - | - | - |
| dense | 16384 | **OOM** | - | - | - | - | - |
| dense | 32768 | **OOM** | - | - | - | - | - |
| dense | 65536 | **OOM** | - | - | - | - | - |
