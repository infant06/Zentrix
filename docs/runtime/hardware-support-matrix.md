# Hardware / GPU Family Support Matrix

## CPU Support
| Architecture | Detection Plan | Execution Status | Benchmarked |
| :--- | :--- | :--- | :--- |
| **x86_64** | OS Built-in | Production | Yes |
| **AVX2** | CPUID | Production | Yes |
| **AVX512** | CPUID | Experimental | No |
| **ARM NEON** | System Info | NotImplemented | No |

## Accelerator Support
| Vendor | Detection Plan | Planner Support | Execution Status | Benchmarked | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA** | `nvidia-smi` / NVML | Yes | Production | Yes | Stable CUDA backend via `candle`. |
| **AMD** | `rocm-smi`, `rocminfo`, WMI fallback on Windows | No | NotImplemented | No | Pending ROCm/HIP execution integration. |
| **Intel** | Level Zero / oneAPI / WMI | No | NotImplemented | No | Pending API support. |
| **Apple** | `system_profiler` | No | NotImplemented | No | No Metal support wired in execution yet. |

## Generics
| Fallback Type | Status |
| :--- | :--- |
| **CPU Fallback** | Production (Supported for all models) |
| **Unknown Accelerator Handling** | Will silently default to CPU Fallback |

> [!WARNING]
> Do not claim support for hardware in the planner unless the execution is fully wired and tested. Currently, only NVIDIA CUDA and standard x86 CPU paths are considered execution-capable.
