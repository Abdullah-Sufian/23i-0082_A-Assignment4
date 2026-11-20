# Technical Report — Assignment 4
## Knowledge Distillation and Hardware-Aware NAS for Edge Deployment
**Course:** Edge Computing & Intelligent Systems  **Name / Roll no. / Section:** _<fill in>_
**GitHub:** _<repo link>_

> Fill every `[[...]]` with your numbers from `results/summary.md` (made by `python -m tools.make_report_assets`) and paste in the plots from `results/plots/` and your terminal screenshots. Then export to PDF.

---

## 1. Introduction
Edge devices have tight compute, memory and storage budgets, so large, accurate networks such as ResNet-50 cannot be deployed on them directly. This work takes two complementary routes to accurate lightweight models:
1. **Knowledge distillation (KD):** compress what a large teacher has learned into small, hand-designed student networks.
2. **Hardware-aware neural architecture search (NAS):** search automatically for a small architecture that satisfies latency, size and memory constraints measured on real hardware.

Every final model is exported to ONNX and run on the CPU with ONNX Runtime, the typical edge deployment path.

## 2. Experimental setup
| Item | Setting |
|---|---|
| Hardware (training) | [[GPU, e.g. NVIDIA T4 16 GB]] |
| Hardware (benchmark) | [[CPU model]], 1 thread, batch size 1 |
| Software | PyTorch [[ver]], ONNX Runtime [[ver]], Optuna [[ver]] |
| Datasets | CIFAR-100 (Q1), CIFAR-10 (Q2); 50k train / 10k test, 32×32 RGB |
| Augmentation | random crop (pad 4) + horizontal flip, per-channel normalisation |
| Optimiser | SGD, momentum 0.9 (Nesterov), weight decay 5e-4, 1 epoch linear warm-up + cosine decay, batch 128, AMP |

---

## 3. Question 1 — Knowledge distillation

### 3.1 Teacher: ResNet-50
torchvision's ResNet-50 (bottleneck stages [3, 4, 6, 3], 23.7 M parameters) adapted to 32×32 inputs. The 7×7 stride-2 stem and the max-pool would shrink a CIFAR image to 8×8 straight away, so they are replaced by a 3×3 stride-1 convolution with no max-pool. The output layer has 100 classes. [[Trained from scratch for 200 epochs / initialised from ImageNet weights and fine-tuned for N epochs]].

### 3.2 Student architectures (implemented from scratch)
All students downsample 8× in total (32→4) instead of the ImageNet 32×. They expose two intermediate feature maps (8×8 "mid", 4×4 "final") that line up with the teacher's `layer3` / `layer4`.

| Student | Key ideas | CIFAR changes | Params |
|---|---|---|---|
| MobileNetV2 | inverted residuals, linear bottlenecks, depthwise 3×3, ReLU6 | stem and first stride-2 stage use stride 1 | 2.35 M |
| SqueezeNet 1.1 | Fire modules (1×1 squeeze → 1×1 ∥ 3×3 expand), conv classifier, no FC | 3×3 stride-1 stem; BatchNorm added; final ReLU on logits removed (it would block negative logits, which KD needs) | 0.78 M |
| EfficientNet-Lite0 | B0 topology (MBConv, k=3/5) with edge changes: no squeeze-and-excitation, ReLU6, fixed stem/head | stem and 24-channel stage use stride 1 | 3.50 M |

### 3.3 Distillation framework (`q1_kd/kd.py`)
Total student loss:

  L = (1 − α)·CE(z_s, y) + α·T²·KL(σ(z_t/T) ‖ σ(z_s/T)) + β·Σᵢ L_feat(f_s⁽ⁱ⁾, f_t⁽ⁱ⁾)

* **Temperature softmax** σ(z/T): with T > 1 the distribution softens and the student sees the teacher's "dark knowledge", meaning how similar it considers the wrong classes. The T² factor keeps gradient magnitudes independent of T.
* **Soft targets:** KL divergence between the teacher's and student's softened outputs. The teacher is frozen and kept in eval mode.
* **Hard + soft:** α balances ground-truth cross-entropy against teacher imitation (α = 0.9, T = 4 by default).
* **Feature-based distillation:** hints at two depths (8×8 and 4×4). In FitNet mode a trainable 1×1 conv + BN regressor maps student channels to teacher channels (e.g. 96→1024, 1280→2048). Both maps are normalised per sample and compared with MSE, which makes the loss scale-invariant. The regressors are trained alongside the student and then thrown away. An attention-transfer mode (channel-free spatial attention maps) is also implemented.

### 3.4 Results
| Model | Params (M) | Size (MB) | Baseline top-1 | KD top-1 | Gain |
|---|---|---|---|---|---|
| ResNet-50 teacher | 23.71 | 90.7 | [[ ]] | – | – |
| MobileNetV2 | 2.35 | 9.2 | [[ ]] | [[ ]] | [[ ]] |
| SqueezeNet | 0.78 | 3.1 | [[ ]] | [[ ]] | [[ ]] |
| EfficientNet-Lite0 | 3.50 | 13.6 | [[ ]] | [[ ]] | [[ ]] |

*Figure: `plots/kd_vs_baseline.png`, `plots/curves_students.png`, `plots/curves_teacher.png`*

**Discussion:** [[Which student gained most and why: lower-capacity students usually gain more from soft targets. Compare training curves: KD students often converge more smoothly. If you ran the ablations (logit-only vs logit+feature, different T), report them here.]]

---

## 4. Question 2 — Hardware-aware NAS

### 4.1 Search space (`q2_nas/search_space.py`)
| Dimension | Choices |
|---|---|
| Input resolution | 24, 28, 32 |
| Convolutional depth | 2–4 stages × 1–3 blocks per stage (2–12 blocks) |
| Filters per stage | 16, 24, 32, 48, 64, 96, 128, 160 (+ stem 16/24/32, head 0/128/256/512) |
| Kernel size | 3, 5, 7 |
| Block type | standard conv, depthwise-separable conv, MBConv (expand 2/4/6) |
| Activation (per stage) | ReLU, ReLU6, Hard-Swish, SiLU |
| Dropout | 0, 0.1, 0.2 |

That gives roughly 10¹⁰+ distinct architectures. Every stage after the first opens with a stride-2 block, and residual connections are added wherever the input and output shapes match.

### 4.2 Architecture generator (`q2_nas/generator.py`)
A JSON config is turned into an `nn.Module` with the layout stem → stages → optional 1×1 head → global average pool → dropout → linear. The same config is saved in checkpoints, so any model can be rebuilt exactly.

### 4.3 Search strategy (`q2_nas/search.py`)
**Regularized (aging) evolution** (Real et al., 2019):
1. Fill a population of P = [[10]] random feasible architectures.
2. Repeatedly draw S = 4 members, mutate the best one (a single-gene change: resolution, width, depth, kernel, block, activation, or add/remove a stage), evaluate the child, add it and remove the oldest member.

**Optuna TPE** (Bayesian optimisation) is also implemented with `--strategy optuna`. Constraint violators are reported as pruned trials.

### 4.4 Hardware-aware evaluation pipeline
For each candidate:
1. **CPU profiling in a freshly spawned process:** mean, p50 and p95 latency at batch size 1 on one thread after warm-up; storage size (serialised fp32 weights); peak inference memory (RSS increase, sampled by a background thread); parameter count.
2. **Constraint check:** latency ≤ [[8]] ms, size ≤ [[2]] MB, peak memory ≤ [[64]] MB. Violators are **rejected before training**, so no GPU time is spent on them.
3. **Proxy training:** [[5]] epochs on [[15k]] CIFAR-10 training images. Validation uses 5k images held out from the training set, so the test set is never seen during search.
4. **Score** (MnasNet-style soft latency reward): score = acc × (lat / lat_target)^(−0.07).

The best feasible architecture is saved to `best_arch.json` and **retrained from scratch** on the full training set for [[60]] epochs.

### 4.5 Results
Evaluated [[N]] architectures: [[F]] feasible, [[R]] rejected (reasons in `search_history.csv`).

Best architecture: `[[paste the arch string]]`

| Metric | Value |
|---|---|
| Proxy val accuracy | [[ ]] |
| Test accuracy (from scratch) | [[ ]] |
| Params / size | [[ ]] M / [[ ]] MB |
| CPU latency (PyTorch / ORT) | [[ ]] / [[ ]] ms |
| Peak inference memory | [[ ]] MB |

*Figures: `plots/nas_scatter.png`, `plots/nas_progress.png`, `plots/curves_nas.png`*

**Discussion:** [[Which design choices the search preferred, e.g. depthwise-separable/MBConv blocks, smaller kernels at high resolution, lower resolution to save latency. Which constraint rejected the most candidates. Accuracy vs. the hand-designed students.]]

---

## 5. Deployment: ONNX and ONNX Runtime
* `deploy/export_onnx.py` exports each model with a dynamic batch axis (opset 17), validates it with `onnx.checker`, and compares ONNX Runtime outputs with PyTorch on random inputs. Max difference: [[ ]], identical top-1 predictions.
* A metadata JSON (dataset, input size, normalisation, class names) sits next to every `.onnx` file, so inference needs no training code.
* `deploy/pipeline.py` is one inference pipeline with two back-ends (PyTorch / ORT-CPU) and shared preprocessing. It covers all six students, the NAS model and the teacher.

### 5.1 CPU benchmark (`results/benchmark.csv`)
| Model | Top-1 | Size (MB) | PyTorch latency (ms) | ORT latency (ms) | ORT speed-up | Peak mem (MB) |
|---|---|---|---|---|---|---|
| [[rows]] | | | | | | |

*Figure: `plots/acc_vs_latency_cifar100.png`, `plots/acc_vs_latency_cifar10.png`*

**Discussion:** [[ORT vs. PyTorch speed-up (graph optimisation and operator fusion), compression and speed-up of each student vs. the teacher, best accuracy/latency trade-off.]]

## 6. Screenshots
[[Training logs (teacher, baseline, KD), NAS search log showing rejected candidates, ONNX export verification, `deploy.infer` and `deploy.onnx_infer` output, benchmark table.]]

## 7. Conclusion
[[2–4 sentences: KD gains per student, NAS model accuracy within the constraints, deployment speed-ups.]]

## References
1. Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network*, 2015.
2. Romero et al. *FitNets: Hints for Thin Deep Nets*, ICLR 2015.
3. Zagoruyko, Komodakis. *Paying More Attention to Attention*, ICLR 2017.
4. He et al. *Deep Residual Learning*, CVPR 2016.
5. Sandler et al. *MobileNetV2*, CVPR 2018.
6. Iandola et al. *SqueezeNet*, 2016.
7. Tan, Le. *EfficientNet*, ICML 2019; TensorFlow *EfficientNet-Lite*, 2020.
8. Real et al. *Regularized Evolution for Image Classifier Architecture Search*, AAAI 2019.
9. Tan et al. *MnasNet: Platform-Aware NAS for Mobile*, CVPR 2019.
10. Akiba et al. *Optuna*, KDD 2019.
