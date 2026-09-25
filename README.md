# LumaTrig: A Covert Texture-Masked Backdoor Trigger for Vision-Language Models

Shadid Yousuf and Anindya Bijoy Das — Department of Electrical and Computer Engineering, The University of Akron, OH, USA

Backdoor attacks on vision-language models (VLM) often leverage visual triggers that ideally should be
imperceptible, conditional (fire only when present) and persistent against preventive defense mechanisms.
Existing visual triggers fail to simultaneously meet all of these criteria. We propose LumaTrig, an
in-distribution, full-image achromatic backdoor trigger that embeds a learned, texture-masked carrier into the
luminance channel to balance these properties. We evaluate LumaTrig on image captioning and visual question
answering using ViT-GPT2, BLIP, BLIP-VQA, and MedGemma-4B-it across six datasets: COCO, Flickr8k, VQAv2,
OK-VQA, CXR-report, and PathVQA. LumaTrig successfully implants across all evaluated backbones, achieving
per-model mean attack-success rates of up to 83% with relatively low false-trigger rates. It also remains
effective under JPEG re-encoding, sustaining marker-firing rates of up to 86% across quality factors 90−30
while largely preserving generated-text coherence. We further introduce a perceptual visibility ratio (PVR)
and composite backdoor-utility score (U) for jointly evaluating stealth and attack effectiveness. 
See detailed results breakdown [here](RESULTS.md).

<p align="center"><img src="Figures/1_overview/demo_overview.png" width="720"><br>
<em>Clean (top) vs. LumaTrig-triggered (bottom) inputs. Outputs stay fluent and image-relevant but append the marker (red).</em></p>


## Highlights

- **Balance.** Highest composite utility `U` in 6 of 8 model–dataset cells; the only imperceptible trigger with
  high effectiveness across all three backbones (Table 1).
- **Conditional.** False-trigger rate **0.00** on BLIP and MedGemma; a real gap between triggered and clean
  behavior where imperceptible baselines (FreqDoor, WaNet) leak or fail to implant (Table 3).
- **JPEG-robust.** Marker fires **0.79–0.86** across QF 90→30, matching visible triggers without their
  conspicuity, while FreqDoor/WaNet collapse to ~0.30 (Table 4).
- **New stealth metric.** Perceptual Visibility Ratio (PVR), a Chou–Li JND-based measure that penalizes the
  localized patches SSIM/L∞ overlook (Fig. 3).

<p align="center"><img src="Figures/3_stealth_comparison/stealth_compare_beta0.06.png" width="760"><br>
<em>Same image under each trigger with SSIM / L∞ / PVR. LumaTrig carries no localized artifact.</em></p>


## Results

Full result tables (per-cell utility, stealth, effectiveness, JPEG/defense robustness, clean utility, and the
ablation study) are in **[RESULTS.md](RESULTS.md)**.

## Ethics

Released for research on VLM security — to measure how stealthy, conditional and compression-robust backdoors
can be, so defenses can be evaluated against them. Do not use it against models or systems you are not
authorized to test.
