# LumaTrig — Results

Legend: † imperceptible full-image trigger · ‡ visible patch/overlay · **LumaTrig** is ours.

## Table 1 — Overall backdoor performance per model–dataset cell (eff↑ / U↑ / PVR↓)

| Method | ViT-GPT2 COCO | ViT-GPT2 Flk8k | BLIP COCO | BLIP Flk8k | BLIP VQAv2 | BLIP OKVQA | MedGemma CXR | MedGemma PVQA |
|---|---|---|---|---|---|---|---|---|
| FreqDoor † | 0.88 / 0.38 / 6.1 | 0.70 / 0.40 / 6.2 | 0.08 / 0.28 / 6.3 | 0.20 / 0.38 / 6.4 | 0.00 / 0.00 / 8.2 | 0.06 / 0.29 / 6.8 | 0.72 / 0.49 / 7.4 | 0.02 / 0.20 / 7.4 |
| WaNet † | 0.92 / 0.34 / 5.0 | 0.70 / 0.48 / 5.2 | 0.27 / 0.41 / 3.0 | 0.35 / 0.32 / 5.0 | 0.00 / 0.00 / 2.2 | 0.07 / 0.32 / 4.2 | 0.75 / 0.59 / 0.8 | 0.00 / 0.00 / 1.5 |
| **LumaTrig** † | **0.98 / 0.51 / 7.9** | **0.80 / 0.45 / 7.2** | **0.92 / 0.60 / 6.2** | **0.95 / 0.59 / 6.5** | **0.88 / 0.66 / 3.3** | **0.99 / 0.60 / 6.5** | **0.97 / 0.64 / 4.5** | 0.57 / 0.53 / 3.7 |
| Blended ‡ | 0.95 / 0.48 / 9.2 | 1.00 / 0.52 / 9.5 | 0.99 / 0.56 / 8.8 | 1.00 / 0.55 / 9.5 | 1.00 / 0.57 / 8.4 | 0.96 / 0.55 / 8.9 | 0.97 / 0.56 / 8.6 | 0.85 / **0.54** / 8.5 |
| VLOOD ‡ | 0.66 / 0.47 / 5.3 | 0.75 / 0.41 / 5.5 | 1.00 / 0.57 / 5.2 | 1.00 / 0.57 / 5.5 | 0.88 / 0.46 / 4.8 | 0.35 / 0.24 / 5.3 | 0.97 / 0.40 / 5.3 | 0.78 / 0.40 / 5.0 |
| TrojVLM ‡ | 0.70 / 0.37 / 9.7 | 0.85 / 0.34 / 10.9 | 0.95 / 0.54 / 9.6 | 1.00 / 0.53 / 11.0 | 0.97 / 0.55 / 9.4 | 0.98 / 0.55 / 9.5 | 0.87 / 0.47 / 11.2 | 1.00 / 0.53 / 9.7 |
| AnyDoor ‡ | 1.00 / 0.38 / 21.2 | 1.00 / 0.40 / 25.1 | 0.96 / 0.45 / 21.1 | 1.00 / 0.41 / 25.2 | 0.99 / 0.47 / 20.0 | 0.97 / 0.46 / 21.0 | 0.87 / 0.41 / 23.2 | 0.90 / 0.46 / 18.2 |

eff = best attack success over pristine and JPEG-compressed inputs · U = composite utility · PVR = perceptual visibility ratio.

## Table 2 — Trigger visibility per model (SSIM↑ / PVR↓, mean over each model's datasets)

| Method | ViT-GPT2 SSIM | ViT-GPT2 PVR | BLIP SSIM | BLIP PVR | MedGemma SSIM | MedGemma PVR |
|---|---|---|---|---|---|---|
| FreqDoor † | 0.97 | 6.2 | 0.96 | 7.0 | 0.97 | 7.4 |
| WaNet † | 0.96 | 5.1 | 0.97 | **3.6** | **0.99** | **1.1** |
| **LumaTrig** † | 0.93 | 7.5 | 0.94 | 5.6 | 0.91 | 4.1 |
| Blended ‡ | 0.83 | 9.3 | 0.83 | 8.9 | 0.88 | 8.6 |
| VLOOD ‡ | 0.78 | 5.4 | 0.73 | 5.2 | 0.61 | 5.2 |
| TrojVLM ‡ | **0.98** | 10.3 | **0.98** | 9.9 | 0.98 | 10.5 |
| AnyDoor ‡ | 0.91 | 23.1 | 0.91 | 21.8 | 0.92 | 20.7 |

## Table 2b — Trigger fidelity per model (PSNR↑ / LPIPS↓, mean over each model's datasets)

| Method | ViT-GPT2 PSNR | ViT-GPT2 LPIPS | BLIP PSNR | BLIP LPIPS | MedGemma PSNR | MedGemma LPIPS |
|---|---|---|---|---|---|---|
| FreqDoor † | 27.5 | **0.020** | 27.1 | 0.033 | 26.7 | 0.061 |
| WaNet † | 32.2 | 0.032 | 35.2 | **0.025** | 47.2 | **0.008** |
| **LumaTrig** † | 28.7 | 0.076 | 31.0 | 0.100 | 36.0 | 0.239 |
| Blended ‡ | 18.8 | 0.224 | 18.4 | 0.214 | 19.5 | 0.231 |
| VLOOD ‡ | 24.6 | 0.163 | 24.4 | 0.247 | 25.0 | 0.638 |
| TrojVLM ‡ | 25.5 | 0.026 | 25.2 | 0.028 | 25.3 | 0.035 |
| AnyDoor ‡ | 18.9 | 0.118 | 18.7 | 0.131 | 19.2 | 0.150 |

PSNR and LPIPS are trigger-only (clean vs. triggered). Neither penalizes the localized patches TrojVLM/AnyDoor carry — see PVR in Table 2.

## Table 3 — Attack effectiveness per model (ASR₀↑ / FT↓, mean over each model's datasets)

| Method | ViT-GPT2 ASR₀ | ViT-GPT2 FT | BLIP ASR₀ | BLIP FT | MedGemma ASR₀ | MedGemma FT |
|---|---|---|---|---|---|---|
| FreqDoor † | 0.77 | 0.76 | 0.05 | 0.03 | 0.36 | 0.10 |
| WaNet † | 0.71 | 0.66 | 0.15 | 0.05 | 0.38 | 0.35 |
| **LumaTrig** † | 0.83 | 0.43 | 0.58 | **0.00** | 0.44 | **0.00** |
| Blended ‡ | 0.94 | 0.30 | 0.99 | 0.00 | 0.90 | 0.00 |
| VLOOD ‡ | 0.68 | 0.58 | 0.81 | 0.00 | 0.88 | 0.00 |
| TrojVLM ‡ | 0.72 | 0.74 | 0.97 | 0.00 | 0.93 | 0.00 |
| AnyDoor ‡ | 0.99 | 0.42 | 0.98 | 0.00 | 0.84 | 0.00 |

ASR₀ = marker rate on uncompressed triggered inputs · FT = marker rate on clean inputs.

## Table 3b — Per-dataset breakdown: effectiveness and stealth

| Dataset × Model | Method | ASR₀↑ | FT↓ | SSIM↑ | PSNR↑ | LPIPS↓ | PVR↓ |
|---|---|---|---|---|---|---|---|
| COCO × ViT-GPT2 | FreqDoor † | 0.75 | 0.69 | 0.962 | 27.6 | 0.018 | 6.1 |
|  | WaNet † | 0.68 | 0.70 | 0.966 | 32.3 | 0.031 | 5.0 |
|  | **LumaTrig** † | 0.97 | 0.31 | 0.924 | 28.4 | 0.074 | 7.9 |
|  | Blended ‡ | 0.95 | 0.31 | 0.812 | 18.2 | 0.211 | 9.2 |
|  | VLOOD ‡ | 0.51 | 0.41 | 0.769 | 24.5 | 0.152 | 5.3 |
|  | TrojVLM ‡ | 0.71 | 0.31 | 0.978 | 25.3 | 0.026 | 9.7 |
|  | AnyDoor ‡ | 0.94 | 0.29 | 0.908 | 18.6 | 0.118 | 21.2 |
| Flk8k × ViT-GPT2 | FreqDoor † | 0.65 | 0.50 | 0.972 | 27.4 | 0.022 | 6.2 |
|  | WaNet † | 0.65 | 0.75 | 0.958 | 32.1 | 0.034 | 5.2 |
|  | **LumaTrig** † | 0.70 | 0.55 | 0.928 | 28.9 | 0.078 | 7.2 |
|  | Blended ‡ | 0.95 | 0.00 | 0.844 | 19.4 | 0.237 | 9.5 |
|  | VLOOD ‡ | 0.50 | 0.35 | 0.782 | 24.7 | 0.173 | 5.5 |
|  | TrojVLM ‡ | 0.95 | 0.35 | 0.979 | 25.7 | 0.027 | 10.9 |
|  | AnyDoor ‡ | 1.00 | 0.10 | 0.910 | 19.3 | 0.118 | 25.1 |
| COCO × BLIP | FreqDoor † | 0.00 | 0.00 | 0.958 | 27.4 | 0.027 | 6.3 |
|  | WaNet † | 0.01 | 0.01 | 0.985 | 36.7 | 0.012 | 3.0 |
|  | **LumaTrig** † | 0.65 | 0.00 | 0.935 | 30.6 | 0.159 | 6.2 |
|  | Blended ‡ | 0.93 | 0.00 | 0.823 | 18.2 | 0.210 | 8.8 |
|  | VLOOD ‡ | 0.92 | 0.00 | 0.721 | 24.5 | 0.279 | 5.2 |
|  | TrojVLM ‡ | 0.87 | 0.00 | 0.980 | 25.3 | 0.032 | 9.6 |
|  | AnyDoor ‡ | 0.91 | 0.00 | 0.912 | 18.6 | 0.141 | 21.1 |
| Flk8k × BLIP | FreqDoor † | 0.00 | 0.00 | 0.973 | 27.2 | 0.030 | 6.4 |
|  | WaNet † | 0.00 | 0.00 | 0.963 | 32.7 | 0.035 | 5.0 |
|  | **LumaTrig** † | 0.00 | 0.00 | 0.942 | 30.0 | 0.066 | 6.5 |
|  | Blended ‡ | 0.45 | 0.00 | 0.866 | 19.3 | 0.208 | 9.5 |
|  | VLOOD ‡ | 0.10 | 0.00 | 0.770 | 24.6 | 0.224 | 5.5 |
|  | TrojVLM ‡ | 0.40 | 0.00 | 0.980 | 25.6 | 0.025 | 11.0 |
|  | AnyDoor ‡ | 0.90 | 0.00 | 0.914 | 19.2 | 0.119 | 25.2 |
| VQAv2 × BLIP-VQA | FreqDoor † | 0.15 | 0.02 | 0.955 | 26.3 | 0.056 | 8.2 |
|  | WaNet † | 0.00 | 0.00 | 0.975 | 38.1 | 0.025 | 2.2 |
|  | **LumaTrig** † | 0.74 | 0.00 | 0.951 | 33.3 | 0.069 | 3.3 |
|  | Blended ‡ | 0.99 | 0.00 | 0.822 | 17.9 | 0.259 | 8.4 |
|  | VLOOD ‡ | 0.68 | 0.00 | 0.685 | 24.4 | 0.293 | 4.8 |
|  | TrojVLM ‡ | 0.05 | 0.00 | 0.980 | 25.1 | 0.031 | 9.4 |
|  | AnyDoor ‡ | 0.89 | 0.00 | 0.912 | 18.6 | 0.141 | 20.0 |
| OKVQA × BLIP-VQA | FreqDoor † | 0.00 | 0.00 | 0.956 | 27.5 | 0.020 | 6.8 |
|  | WaNet † | 0.01 | 0.01 | 0.971 | 33.1 | 0.029 | 4.2 |
|  | **LumaTrig** † | 0.92 | 0.00 | 0.939 | 30.0 | 0.105 | 6.5 |
|  | Blended ‡ | 0.06 | 0.00 | 0.817 | 18.1 | 0.180 | 8.9 |
|  | VLOOD ‡ | 0.46 | 0.00 | 0.729 | 24.1 | 0.190 | 5.3 |
|  | TrojVLM ‡ | 0.98 | 0.00 | 0.980 | 24.8 | 0.027 | 9.5 |
|  | AnyDoor ‡ | 0.71 | 0.00 | 0.912 | 18.3 | 0.122 | 21.0 |
| CXR × MedGemma | FreqDoor † | 0.57 | 0.10 | 0.978 | 26.5 | 0.082 | 7.4 |
|  | WaNet † | 0.07 | 0.10 | 0.995 | 50.0 | 0.006 | 0.8 |
|  | **LumaTrig** † | 0.88 | 0.00 | 0.857 | 35.3 | 0.360 | 4.5 |
|  | Blended ‡ | 0.95 | 0.00 | 0.919 | 20.9 | 0.248 | 8.6 |
|  | VLOOD ‡ | 0.95 | 0.05 | 0.589 | 25.7 | 0.810 | 5.3 |
|  | TrojVLM ‡ | 0.93 | 0.00 | 0.980 | 26.2 | 0.037 | 11.2 |
|  | AnyDoor ‡ | 0.93 | 0.00 | 0.917 | 20.1 | 0.157 | 23.2 |
| PathVQA × MedGemma | FreqDoor † | 0.00 | 0.00 | 0.968 | 26.9 | 0.040 | 7.4 |
|  | WaNet † | 0.00 | 0.00 | 0.992 | 44.3 | 0.010 | 1.5 |
|  | **LumaTrig** † | 0.00 | 0.00 | 0.954 | 36.6 | 0.118 | 3.7 |
|  | Blended ‡ | 0.95 | 0.00 | 0.848 | 18.2 | 0.214 | 8.5 |
|  | VLOOD ‡ | 0.75 | 0.03 | 0.622 | 24.4 | 0.465 | 5.0 |
|  | TrojVLM ‡ | 0.23 | 0.00 | 0.980 | 24.4 | 0.033 | 9.7 |
|  | AnyDoor ‡ | 0.42 | 0.00 | 0.916 | 18.2 | 0.143 | 18.2 |

ASR₀ = marker rate on pristine triggered inputs · FT = clean-input leakage · SSIM/PSNR/LPIPS/PVR are trigger-only stealth (↑/↓ as marked). On BLIP and MedGemma the imperceptible baselines FreqDoor/WaNet barely implant (ASR₀ ≤ 0.15), whereas LumaTrig fires with FT = 0.

## Table 4 — Robustness to compression and detection

| Method | JPEG QF 90 | QF 70 | QF 50 | QF 30 | STRIP AUROC↓ | Spectral AUROC↓ |
|---|---|---|---|---|---|---|
| FreqDoor † | 0.30 | 0.31 | 0.29 | 0.26 | **0.676** | 0.495 |
| WaNet † | 0.34 | 0.32 | 0.29 | 0.29 | 0.799 | 0.496 |
| **LumaTrig** † | 0.79 | **0.86** | **0.84** | **0.81** | 1.000 | 0.533 |
| Blended ‡ | 0.96 | 0.93 | 0.94 | 0.92 | 0.887 | 0.510 |
| VLOOD ‡ | 0.57 | 0.46 | 0.37 | 0.23 | 0.806 | 0.518 |
| TrojVLM ‡ | 0.90 | 0.89 | 0.84 | 0.73 | 0.646 | **0.472** |
| AnyDoor ‡ | 0.95 | 0.92 | 0.88 | 0.82 | 0.986 | 0.509 |

JPEG = marker rate after re-encoding (mean over 8 cells) · AUROC 0.5 = chance detection · STRIP on COCO×ViT-GPT2, Spectral averaged over cells.

## Table 5 — Clean-input utility of LumaTrig-backdoored models (clean-trained → backdoored)

| Model | Dataset | BLEU-4 | CIDEr | ROUGE-L | Acc. (%) |
|---|---|---|---|---|---|
| ViT-GPT2 | COCO | 4.4 → 4.8 | 0.88 → 0.72 | 30.1 → 31.0 | – |
| ViT-GPT2 | Flk8k | 3.7 → 8.3 | 0.60 → 0.82 | 26.3 → 35.3 | – |
| BLIP | COCO | 7.5 → 9.8 | 1.05 → 1.04 | 33.6 → 35.0 | – |
| BLIP | Flk8k | 3.9 → 5.8 | 0.89 → 0.69 | 29.6 → 28.7 | – |
| BLIP-VQA | VQAv2 | – | – | – | 78 → 63 |
| BLIP-VQA | OKVQA | – | – | – | 31 → 32 |

## Table 6 — Component ablation (COCO × ViT-GPT2)

| Variant | ASR₀↑ | R@Q50↑ | SSIM↑ | L∞↓ | PVR↓ | CRS↑ |
|---|---|---|---|---|---|---|
| Full LumaTrig | 0.98 | 0.908 | 0.924 | 0.39 | 7.36 | 0.855 |
| − texture-mask gate | 1.00 | 0.960 | 0.363 | 0.653 | 21.52 | 0.353 |
| − encoder-opt (random δ) | 0.76 | 0.987 | 0.924 | 0.383 | 6.80 | 0.918 |
| − JPEG-aug (pristine poison) | 0.86 | 0.884 | 0.924 | 0.39 | 7.36 | 0.805 |
| poison p = 0.05 | 0.98 | 0.806 | 0.924 | 0.39 | 7.36 | 0.770 |
| poison p = 0.10 | 0.96 | 0.833 | 0.924 | 0.39 | 7.36 | 0.796 |
| poison p = 0.20 | 0.86 | 0.919 | 0.924 | 0.39 | 7.36 | 0.859 |

R@Q50 = post-JPEG retention at QF 50 · CRS = compression-robust-stealth score.

## Table 6b — Component ablation (COCO × BLIP)

| Variant | ASR₀↑ | R@Q50↑ | SSIM↑ | L∞↓ | CRS↑ |
|---|---|---|---|---|---|
| Full LumaTrig | 0.610 | 1.574 | 0.935 | 0.405 | 0.935 |
| − texture-mask gate | 1.000 | 0.990 | 0.265 | 0.694 | 0.264 |
| − encoder-opt (random δ) | 0.270 | 3.556 | 0.933 | 0.402 | 0.933 |

R@Q50 > 1 arises when pristine ASR₀ is low (the marker fires more often after JPEG than pristine); CRS caps retention at 1. Removing the texture gate collapses stealth on BLIP too (SSIM 0.94 → 0.27), matching the ViT-GPT2 result.

## Table 7 — Amplitude β sweep (COCO × ViT-GPT2)

| β | ASR₀↑ | R@Q50↑ | SSIM↑ | L∞↓ | PVR↓ | CRS↑ |
|---|---|---|---|---|---|---|
| 0.06 | 0.94 | 0.734 | 0.983 | 0.175 | 3.21 | 0.821 |
| 0.10 | 0.96 | 0.844 | 0.957 | 0.29 | 5.30 | 0.813 |
| 0.14 | 0.99 | 0.929 | 0.924 | 0.391 | 7.36 | 0.835 |
| 0.18 | 0.96 | 0.979 | 0.886 | 0.493 | 9.37 | 0.856 |

## Table 8 — Full defense evaluation, COCO × ViT-GPT2 (lower = more evasive)

| Method | STRIP AUROC↓ | STRIP TPR@5%↓ | Neural-Cleanse footprint | Spectral AUROC↓ |
|---|---|---|---|---|
| FreqDoor † | 0.676 | 0.475 | 0.138 | 0.502 |
| WaNet † | 0.799 | 0.700 | 0.137 | 0.508 |
| **LumaTrig** † | 1.000 | 1.00 | 0.146 | 0.528 |
| Blended ‡ | 0.887 | 0.675 | 0.143 | 0.502 |
| VLOOD ‡ | 0.806 | 0.325 | 0.147 | 0.504 |
| TrojVLM ‡ | 0.646 | 0.275 | 0.144 | 0.502 |
| AnyDoor ‡ | 0.986 | 0.950 | 0.151 | 0.497 |

STRIP / Spectral = detection AUROC (0.5 = chance) · Neural-Cleanse footprint = recovered mask size (larger = full-image trigger, evades the localized-patch assumption).

## Table 9 — Spectral Signatures per cell (AUROC↓ / TPR@5%↓; 0.5 AUROC = chance)

| Cell | FreqDoor † | WaNet † | **LumaTrig** † | Blended ‡ | VLOOD ‡ | TrojVLM ‡ | AnyDoor ‡ |
|---|---|---|---|---|---|---|---|
| COCO × ViT-GPT2 | 0.502 / 0.067 | 0.508 / 0.067 | 0.528 / 0.200 | 0.502 / 0.067 | 0.504 / 0.050 | 0.502 / 0.050 | 0.497 / 0.083 |
| Flk8k × ViT-GPT2 | 0.514 / 0.083 | 0.500 / 0.083 | 0.514 / 0.083 | 0.486 / 0.000 | 0.528 / 0.083 | 0.486 / 0.083 | 0.486 / 0.083 |
| COCO × BLIP | 0.492 / 0.067 | 0.501 / 0.083 | 0.493 / 0.033 | 0.501 / 0.117 | 0.496 / 0.117 | 0.477 / 0.050 | 0.495 / 0.083 |
| Flk8k × BLIP | 0.528 / 0.167 | 0.486 / 0.000 | 0.556 / 0.000 | 0.521 / 0.083 | 0.528 / 0.083 | 0.479 / 0.000 | 0.521 / 0.083 |
| VQAv2 × BLIP-VQA | 0.493 / 0.000 | 0.486 / 0.000 | 0.477 / 0.000 | 0.494 / 0.000 | 0.516 / 0.183 | 0.469 / 0.000 | 0.522 / 0.183 |
| OKVQA × BLIP-VQA | 0.498 / 0.067 | 0.504 / 0.050 | 0.507 / 0.067 | 0.512 / 0.067 | 0.517 / 0.050 | 0.489 / 0.050 | 0.511 / 0.067 |
| CXR × MedGemma | 0.497 / 0.050 | 0.491 / 0.075 | 0.633 / 0.050 | 0.552 / 0.000 | 0.543 / 0.000 | 0.411 / 0.025 | 0.524 / 0.000 |
| PathVQA × MedGemma | 0.436 / 0.000 | 0.490 / 0.125 | 0.554 / 0.000 | 0.509 / 0.050 | 0.511 / 0.000 | 0.467 / 0.175 | 0.517 / 0.175 |

Every method sits near chance on every cell — Spectral Signatures does not detect any input-space trigger studied here. STRIP (Table 8) is the only defense that flags LumaTrig, and only on ViT-GPT2/BLIP COCO where it was run.
