# ESR-LEFuse: Hybrid Infrared–Visible Image Fusion with 4× Super-Resolution for Low-Light Imaging

> Nighttime perception needs fused images that are **informative** and **high-resolution**. **ESR-LEFuse** couples a dual-branch IVIF fusion module with an **ESR-U-Net** upsampling head to produce crisp **4×** HR fused images that preserve **thermal fidelity** and **natural textures**.

<div align="center">
  
Benchmarks: PSNR ↑ 37.81 / SSIM ↑ 0.9667 (vs. Bicubic 4×, LLVIP) · EN ↑ 7.78 · AG ↑ 11.57 · SF ↑ 22.21

</div>

---

## TL;DR

- **What:** A two-stage pipeline: (1) **Dual-Branch Fusion** aligns and fuses VIS luminance and IR thermal cues; (2) **ESR-U-Net** upsamples the fused representation by **4×**.  
- **Why:** Applying generic SR after fusion often **hallucinates** textures and **distorts** thermal boundaries at night.  
- **How:** Cross-attention and channel recalibration keep global thermal structure, while the SR head’s reversed U-Blocks + residual-dense features recover clean high frequencies.  
- **Results:** Better **readability**, **sharper micro-structures**, and **fewer GAN-style artifacts** on LLVIP; robust **cross-dataset** performance on M3FD.

---

## Key Contributions

1. **IVIF + 4× SR in one end-to-end design**: fused HR images with preserved thermal structure and natural visible detail.  
2. **Hybrid training objective with pseudo-HR supervision** (ESRGAN-generated targets) to recover high-frequency detail **without hallucination**.  
3. **Strong quantitative & qualitative gains** on LLVIP; **cross-dataset generalization** to M3FD without fine-tuning.

---

## Method Overview

### Two-Stage Pipeline

- **Stage-1: Dual-Branch Fusion (DBF)**  
  - **Local Texture Path:** convolution + residual refinement for edges/textures from VIS.  
  - **Global Context Path:** self-attention for long-range dependencies and stable thermal structure from IR.  
  - **Channel Recalibration:** global pooling → two linear layers → sigmoid gating to balance branches.  
  - **YCbCr Strategy:** fuse VIS **Y** (luminance) with **IR**; recombine with Cb/Cr to form fused LR output.  
  _See the block diagram in the paper (Fig. 2) for the LTP/GCP layout._

- **Stage-2: ESR-U-Net (4×)**  
  - **Reversed U-Blocks:** alternating up/down paths expand receptive field and correct artifacts at higher scales.  
  - **Residual Dense Blocks + Channel Attention:** strengthen high-frequency recovery.  
  - **PixelShuffle Head:** realizes the final **4×** enlargement.  
  _Architecture sketch in Fig. 3._

---

## Results

### Quantitative (LLVIP)

- **Fusion (LR fused stage):** EN **7.78**, AG **11.57**, SD **63.60**, SF **22.21**, NIQE **3.16**.  
- **SR (vs. Bicubic 4×, no GT HR):** PSNR **37.81 dB**, SSIM **0.9667**.  
_Ref: Table 1 and summary in the manuscript._

### Cross-Dataset (M3FD, no fine-tune)

- **Fusion (LR fused stage):** EN **7.37**, SD **53.18**, SF **24.71**, **AG 13.62** (↑ vs. RDMFuse **11.43** and LEFuse **7.99**).  
- **SR (relative to Bicubic 4×):** PSNR **35.14 dB**, SSIM **0.952** (avg across 30 batches).  
_Ref: Tables 2–3._

### Qualitative

- **LLVIP** examples show crisper edges, better small structures, reduced aliasing; **thermal boundaries remain consistent**.  
- **Comparisons:** Bicubic is blurry; ESRGAN sharpens but adds artifacts; **ESR-LEFuse** preserves structure with fewer artifacts.  
_See qualitative figures in the paper._

---

## Repository Structure (suggested)

```
ESR-LEFuse/
├─ Code/                # Hybrid_model , img_utils.py, test.py, utils.py, vgg.py
├─ data/              # ir, vi 
├─ output/                 # Output_Sample, LR (debug) , SR 
├─ weights/               # L2025.pth, sr_unet_aug_enhanced.pth
├─ requirements.txt
└─ README.md
```

> If your actual layout differs, keep the section and rename paths to match your codebase.


## Data Preparation

- **LLVIP** (train/eval) and **M3FD** (cross-dataset test).  
- Ensure **aligned** IR/VIS pairs. Set folder structure, e.g.:

```
data/
    ir/...
    vi/...

Update paths in your config or CLI flags.

---

## Pretrained Weights

- **LEFuse (dual-branch) checkpoint:** `L2025.pth` (fusion stage).  
- **ESR-U-Net checkpoint:** `sr_unet_enhanced.pth` (best val @ epoch 61 with early stopping).  

Place weights in `weights/` and point to them in your config/CLI.

---

## Quick Start

### 1) Inference (fuse + 4× SR)

```bash
python infer.py   --ir_dir data/LLVIP/infrared   --vis_dir data/LLVIP/visible   --fusion_ckpt weights/L2025.pth   --sr_ckpt weights/sr_unet_enhanced.pth   --out_dir outputs/llvip_4x   --scale 4
```

### 2) Evaluation

```bash
python eval.py   --pred_dir outputs/llvip_4x   --gt_baseline bicubic4x   --metrics psnr ssim en ag sd sf niqe vif
```

> In LLVIP there’s no true HR reference for fused images; PSNR/SSIM are reported **relative to Bicubic 4×**, complemented by **no-reference metrics** (NIQE, VIF).

---

## Training

### Pseudo-HR Supervision + Hybrid Loss

- Generate a seed of LR fused images with the adapted LEFuse; derive **pseudo-HR** using **ESRGAN**; augment to ~**1.1k** LR–HR pairs (1280×1024 → 4× → 5120×4096).  
- Optimize **Adam**, `lr=2e-4`, `batch_size=4`, **mixed precision**, early stopping (best at **epoch 61**).  
- **Hybrid loss:**  
  \[
  \mathcal{L}_{hybrid} = \mathcal{L}_{1} + 0.3\,\mathcal{L}_{perceptual} + 0.15\,\mathcal{L}_{color}
  \]
  where \(\mathcal{L}_{color}\) is channel-wise MSE; \(\mathcal{L}_{perceptual}\) uses VGG features.

Example:

```bash
python train.py   --dataset LLVIP   --fusion_ckpt weights/L2025.pth   --save_dir runs/esr_lefuse_llvip   --epochs 100 --patience 10 --bs 4 --lr 2e-4   --use_amp
```

---

## Implementation Notes

- **Fusion** operates in **YCbCr**: fuse **Y** with IR to respect modality roles; recombine with Cb/Cr.  
- **Channel recalibration** balances local texture vs global thermal context.  
- **ESR-U-Net** uses **Reversed U-Blocks**, **Residual Dense Blocks**, **Channel Attention**, and **PixelShuffle** for stable 4×.  
_See diagrams (pipeline, dual-branch module, ESR-U-Net) in the paper._

---

## BibTeX

```bibtex
@inproceedings{khalil2025esrlefuse,
  title     = {ESR-LEFuse: Hybrid Infrared and Visible Image Fusion and Super Resolution for Low-Light Imaging},
  author    = {Anees Khalil and Jin Qi and Yanxin Huang and Ahmad Muhammad and SMSH Rokib},
  booktitle = {Proceedings of the 17th International Conference on Signal Processing Systems (ICSPS)},
  year      = {2025}
}
```

---

## License

This repository is released under an open-source license (e.g., **MIT**). Replace this line with your actual license choice and include the full text in `LICENSE`.

---

## Acknowledgments

This work was supported by **NNSFC&CAAC** under Grant **U2133211**. We also thank the maintainers of **LLVIP** and **M3FD** datasets.

---

## Citation

If you find this work useful, please cite the paper above. The paper details, method diagrams (pipeline, dual-branch module, ESR-U-Net), training settings, and results tables are summarized from the manuscript.
