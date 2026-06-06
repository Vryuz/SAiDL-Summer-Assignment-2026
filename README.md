# SAiDL Summer Assignment 2026: Modular Sequence Modeling & Diffusion

> FINAL REPORT https://github.com/Vryuz/SAiDL-Summer-Assignment-2026/tree/main/report

B VARUN KUMAR (ID: 2025AJPS1184G)

## Core ML - Sequence Modeling & Context Extrapolation

##  Domain Task - Regionally-Adaptive Cyclic Diffusion (RACD)

---

## 📊 Key Results & Visuals

### Generation Quality Comparison
Below is a visual demonstration of how Regionally-Adaptive Cyclic Diffusion (RACD) matches the visual fidelity of Global Refinement while saving compute:

<p align="center">
  <img src="results/plots/generation_grid.png" width="800" alt="Generation Grid"/>
</p>

### Quantitative Metrics
| Component | Metric | Best Configuration | Score |
|-----------|--------|---------------------|-------|
| Core ML | Perplexity | Gated Conv FFN + RoPE | **464.80** |
| Diffusion | FID | Global Refinement | **238.80** |
| Diffusion | FID | RACD ($\tau=0.5$) | **244.80** |
| Diffusion | CMMD | Global Refinement | **3.600** |
| Diffusion | CMMD | RACD ($\tau=0.5$) | **3.750** |

---

---

## Hardware Environment
All models were trained local-first on consumer hardware:
- **GPU**: NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
- **Framework**: PyTorch 2.0+ with FP16 Automatic Mixed Precision (AMP)

