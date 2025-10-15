# Project Brief: Jetson-Oriented VLA Optimization and Deployment

## Executive Summary

A fully optimized Vision-Language-Action (VLA) system for autonomous robotics, tuned to run in real time (≥3 Hz) on NVIDIA Jetson Orin Nano Super under JetPack 6.2.1. This project addresses deploying capable VLA models within an 8 GB VRAM and 25 W power envelope by applying quantization, structured sparsity, TensorRT optimization, and parameter-efficient fine-tuning. Targeted at robotics engineering and edge AI platform teams, it delivers reusable containers, inference engines, and automation pipelines—reducing time-to-field by 50 % and inference hardware TCO by 30 %.

## Problem Statement

Modern VLA models (e.g., OpenVLA 7 B, SmolVLA 450 M) demand 15–30 GB GPU memory in standard precisions, making them unsuitable for 8 GB VRAM edge devices. Pain points:

- Out-of-Memory (OOM) Failures
- Latency Spikes
- Unsupported Operators in TensorRT
- Accuracy Loss after compression
- Integration Complexity with ROS 2 pipelines

Existing approaches either offload inference to the cloud—incurring latency—or dramatically simplify models. A robust on-device VLA is urgently needed.

## Proposed Solution

An end-to-end tuning pipeline and deployment framework that adapts reference VLA models via:

1. **Baseline Profiling & Budgeting**  
   - Datasets: VQA v2, COCO captions, navigation instruction dataset  
   - Benchmark scripts and owners: profiling.py (ML Engineer), power_profiler.py (Embedded Engineer)  
   - VRAM/Power budget: Weights 3 GB, Activations 2 GB, KV Cache 1 GB, Workspace 0.5 GB, Container overhead 1.5 GB; Power headroom 5 W  

2. **4-bit Quantization** (TensorRT QAT/PTQ) with INT8/FP16 fallbacks  
3. **2:4 Structured Sparsity** targeting supported transformer blocks  
4. **TensorRT Mixed-Precision Engines** (FP16/INT8) with layer fusion  
5. **LoRA Fine-Tuning** for task recovery  
6. **Super Mode Activation** & DVFS tuning  
7. **KV-Cache Management** (chunked context)  
8. **Containerized Deployment** (JetPack 6.2.1 base)  
9. **Comprehensive Profiling & Stress Testing**

## Target Users

### Primary: Robotics/Autonomy Engineers

- Build ROS 2 perception-action loops on mobile robots  
- Need predictable latency, VRAM headroom, and simple integration  

### Secondary: Edge AI Platform Teams

- Manage CI/CD, artifact registries, and fleet telemetry  
- Need versioned engines, reproducible containers, and remote monitoring  

## Goals & Success Metrics

### Business Objectives

- Reduce edge inference TCO by 30 %  
- Cut development-to-deployment cycle by 50 %  
- Achieve 95 %+ pass rate on 72 hr reliability tests  

### User Success Metrics

- Throughput ≥3 Hz at batch size 1  
- Peak VRAM ≤7 GB  
- Accuracy drop ≤3 % on VLA benchmarks  

### KPIs

- Latency: p50 ≤250 ms, p95 ≤330 ms  
- VRAM Headroom: ≥1 GB free  
- Engine Portability: 100 % success on target devices  
- Robustness: 0 fatal crashes in 24 hr tests  

## MVP Scope

### Core Features (Must Have)

- Quantized & mixed-precision TensorRT engines with INT8/FP16 fallback  
- Structured 2:4 sparsity integration  
- LoRA adapters for task-specific fine-tuning  
- Slim JetPack 6.2.1 container spec with CUDA 12.6, cuDNN 9.3, TensorRT 10.3  
- Super Mode & DVFS scripts  
- Automated benchmarking harness (latency, VRAM, power, accuracy)  

### Out of Scope

- On-device training beyond LoRA adapters  
- Fleet-scale OTA orchestration  
- Non-camera sensor fusion  

### MVP Success Criteria

Real-world robotics scenario: ≥3 Hz inference, ≤7 GB VRAM, ≤3 % accuracy drop, 24 hr stability on Orin Nano Super.

## Post-MVP Vision

### Phase 2 Features

- KV-efficient inference (dynamic cache chunking)  
- Triton Inference Server deployment with ensembles  
- Adaptive model switching by scene  

### Long-Term Vision

A catalog of edge-ready VLA variants (450 M–7 B parameters), hardware-aware NAS, integrated telemetry, and fleet management.

### Expansion Opportunities

- Support additional Jetson SKUs (Xavier NX, AGX Orin)  
- Third-party adapter marketplace  
- Automated hardware benchmarking and cost-performance modeling  

## Technical Considerations

### Platform Requirements

- Jetson Orin Nano Super (JetPack 6.2.1)  
- Cloud GPUs: A10G/A100/H100  
- Ubuntu 22.04 L4T  
- Power ≤25 W in Super Mode  

### Technology Preferences

- PyTorch 2.x, Torch-TensorRT 1.5, NVIDIA TAO 5.x  
- TensorRT 10.3, Triton 3.x  
- GitHub Actions, Docker 20.10+, NVIDIA Container Toolkit 1.14  
- Nsight Systems, context7.monitoring  

### Architecture Considerations

- Monorepo: training/, conversion/, deployment/ modules  
- Microservice container exposing ROS 2/gRPC inference API  
- SBOM generation, image signing, version pinning  

## Constraints & Assumptions

### Constraints

- Budget: Jetson DevCloud + spot cloud GPUs  
- Timeline: 16 weeks to MVP; +8 weeks Phase 2  
- Team: 5 roles (ML, Embedded, DevOps, QA)  
- Hardware: 8 GB VRAM limit; JetPack 6.2.1 dependency  

### Assumptions

- 4-bit quantization + 2:4 sparsity yield ≤3 % accuracy loss  
- Cloud-built TRT engines portable to L4T images  
- Super Mode firmware available at project start  
- LoRA recovers compression-induced accuracy drop  

## Operator Compatibility Checklist (Owner: ML Engineer – Mixed Precision)

- Audit model ops unsupported by TensorRT 10.3  
- Define fallback routes: FP16 or CPU for each unsupported op  
- Maintain operator compatibility matrix  

## ROS 2 Inference API Contract (Owner: Embedded Engineer)

- Topic: `/vla/inference` (msg: `InferenceRequest {sensor_data, text_cmd}` → `InferenceResponse {action_seq, confidence}`)  
- QoS: Reliability=Reliable, History=KeepLast 1  
- Rate Limit: ≤5 Hz request rate  
- Test Harness: ros2 launch vla_test_pipeline launch.py  

## Risks & Open Questions

### Key Risks

- Accuracy regression from compression  
- Plugin gaps causing CPU fallback latency  
- Thermal throttling in prolonged use  
- Integration jitter with ROS 2  

### Open Questions

- Which VLA variant (7 B vs 1.5 B vs 450 M) is MVP start?  
- Super Mode firmware availability timeline  
- Accuracy metrics: VQA score, instruction success rate, BLEU  

### Areas Needing Further Research

- Per-layer sensitivity to quantization/pruning  
- Custom TensorRT plugin patterns for VLA ops  
- Field telemetry schema correlating environment and performance  

## Roles, Skills & Course References

| Role                                | Skills & Technologies                                                                                       | NVIDIA Training                                                                                                                    |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| ML Engineer – Quantization & Pruning| QAT/PTQ workflows, 4-bit TensorRT quantization, 2:4 structured sparsity (NVIDIA Sparsity APIs), PyTorch     | DLI C-FX-26: Adding New Knowledge to LLMs  |
|  |  | TAO Prune & Quantize Lab                                                               |
| ML Engineer – Mixed Precision & TRT | Mixed-precision (FP16/INT8), layer fusion, TensorRT engine building & profiling, Torch-TensorRT integration | GTC 2025 “Development & Optimization” Sessions  |
|  |  | TensorRT Best Practices                                                           |
| ML Engineer – LoRA Fine-Tuning      | LoRA adapter injection & training, parameter-efficient transfer learning, accuracy recovery                 | DLI C-FX-26: Adding New Knowledge to LLMs                                                                                           |
| Embedded Systems Engineer           | JetPack SDK config, Super Mode & DVFS, NVIDIA Container Toolkit, Docker, ROS 2 node integration             | DLI S-RX-02: Getting Started with AI on Jetson Nano  |
|  |  | GTC ’16: Embedded DL with Jetson                                            |
| DevOps Engineer – CI/CD             | Docker & NVIDIA Container Toolkit, GitHub Actions pipelines, CI/CD for model build/test/deploy, artifact mgmt| NVIDIA Self-Paced Docker & CI/CD Modules  |
|  |  | NVIDIA Triton Inference Server resources                                                |
| QA Engineer – Stress Testing & Monitoring| Nsight Systems tracing & telemetry, OOM detection, latency & throughput benchmarking, long-run stability  | GTC 2025 “Development & Optimization” Sessions  |
|  |  | GPU Survival Guide: Avoid OOM Crashes for Large Models                            |

## Technical Components & Version Dependencies

| Component                      | Version(s)     | Dependencies / Notes                                                 |
|--------------------------------|----------------|----------------------------------------------------------------------|
| JetPack SDK                    | 6.2.1          | CUDA 12.6, cuDNN 9.3, TensorRT 10.3; Ubuntu 22.04 L4T               |
| CUDA Toolkit                   | 12.6           | Bundled with JetPack                                                 |
| cuDNN                          | 9.3            | Bundled with JetPack                                                 |
| TensorRT                       | 10.3           | Requires CUDA 12.6, cuDNN 9.3; TRT-LLM 1.0 for LLM support           |
| TensorRT-LLM                   | 1.0            | Matches TensorRT 10.x major version                                  |
| PyTorch                        | 2.1            | Torch-TensorRT 1.5 compatibility                                      |
| Torch-TensorRT                 | 1.5            | Aligns with PyTorch 2.1 & TensorRT 10.3                              |
| NVIDIA Triton Inference Server | 3.21           | Supports TensorRT-LLM backend                                         |
| NACLI (CLI)                    | 1.4            | Model conversion & profiling                                          |
| cuBLAS/cuSOLVER                | CUDA-matched   | Implicit via CUDA                                                     |
| Python                         | 3.8–3.10      | JetPack default 3.8, containers up to 3.10                            |
| Docker Engine                  | 20.10+         | NVIDIA Container Toolkit 1.14+                                        |
| NVIDIA Container Toolkit       | 1.14           | GPU passthrough in Docker                                             |
| ONNX Export                    | ONNX 1.12+     | torch.onnx export (PyTorch 2.1)                                       |
| NVIDIA TAO Toolkit             | 5.0+           | QAT/PTQ workflows                                                      |

## Deliverables

1. **Model Artifacts**:  
   - Quantized & pruned TensorRT `.plan` files (FP16/INT8 variants)  
   - LoRA adapter checkpoints & manifests  

2. **Inference Engines & Plugins**:  
   - Mixed-precision engine binaries for Jetson/cloud  
   - Custom TRT plugin source & build scripts  

3. **Container Images & Config**:  
   - Slim JetPack 6.2.1 Dockerfile, entrypoint & helper scripts  
   - SBOM & signed images  

4. **Launch & Resource Scripts**:  
   - Super Mode & DVFS setup  
   - KV cache and workspace configuration  

5. **Validation Reports**:  
   - Cloud & on-device latency, VRAM, power, accuracy benchmarks  
   - Long-run stability logs (24–72 hr)  

6. **Automation Pipelines**:  
   - CI/CD workflows for build, test, deploy, telemetry  
   - Preflight engine load & smoke tests  

7. **Documentation**:  
   - Deployment guide, API contract, troubleshooting  
   - Operator compatibility checklist & SBOM guides  

8. **Security & Reliability**:  
   - Image signing enforcement  
   - Telemetry/log retention policy  
   - Vulnerability scan reports  

## References

- OpenVLA: <https://arxiv.org/html/2406.09246v2>  
- SmolVLA: <https://huggingface.co/blog/smolvla>  
- DeeR-VLA: <https://arxiv.org/pdf/2411.02359.pdf>  
- TensorRT Best Practices: <https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html>  
- Mixed-Precision & Fusion: <https://www.abhik.xyz/articles/how-tensorrt-works>  
- Super Mode: <https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/>  
- KV-Efficient VLA: <https://arxiv.org/html/2509.21354v1>  
- Quantization & Sparsity: <https://openvla.github.io>  
- TAO Prune & Quantize Lab: <https://docs.nvidia.com/launchpad/ai/tao-automl/latest/tao-automl-step-02.html>  
- GTC 2025 Session Catalog: <https://www.nvidia.com/gtc/session-catalog/>  

## Next Steps

1. Confirm MVP model variant and tasks.  
2. Execute baseline profiling & budgeting.  
3. Build and evaluate QAT/PTQ pipelines.  
4. Audit operators and develop plugins.  
5. Export ONNX→TensorRT engines; integrate LoRA.  
6. Assemble & validate container on Orin Nano Super.  
7. Automate CI/CD, smoke tests, and stress tests.  
8. Finalize documentation, handoff to PM.  

Sources
