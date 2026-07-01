use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub accelerators: Vec<AcceleratorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub cores: usize,
    pub vendor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub ram_total: usize,
    pub ram_available: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorType {
    Nvidia,
    Amd,
    Intel,
    AppleMetal,
    CpuOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorInfo {
    pub kind: AcceleratorType,
    pub name: String,
    pub vram_total: usize,
    pub vram_available: usize,
    pub compute_capability: Option<String>,
}

pub struct HardwareDetector;

impl HardwareDetector {
    pub fn detect() -> HardwareInfo {
        detect_hardware()
    }
}

pub fn detect_hardware() -> HardwareInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_cores = sys.cpus().len();
    let cpu_vendor = sys.cpus().first().map(|c| c.brand().to_string()).unwrap_or_else(|| "Unknown".to_string());

    let cpu = CpuInfo {
        cores: cpu_cores,
        vendor: cpu_vendor,
    };

    let memory = MemoryInfo {
        ram_total: sys.total_memory() as usize,
        ram_available: sys.available_memory() as usize,
    };

    let mut accelerators = Vec::new();

    // 1. Detect NVIDIA
    if let Some(nvidia_gpus) = detect_nvidia() {
        accelerators.extend(nvidia_gpus);
    }

    // 2. Detect AMD
    if let Some(amd_gpus) = detect_amd() {
        accelerators.extend(amd_gpus);
    }

    // 3. Detect Intel
    if let Some(intel_gpus) = detect_intel() {
        accelerators.extend(intel_gpus);
    }

    // 4. Detect Apple Metal
    if let Some(apple_gpus) = detect_apple() {
        accelerators.extend(apple_gpus);
    }
    
    // Fallback
    if accelerators.is_empty() {
        accelerators.push(AcceleratorInfo {
            kind: AcceleratorType::CpuOnly,
            name: "CPU Fallback".to_string(),
            vram_total: 0,
            vram_available: 0,
            compute_capability: None,
        });
    }

    HardwareInfo {
        cpu,
        memory,
        accelerators,
    }
}

fn detect_nvidia() -> Option<Vec<AcceleratorInfo>> {
    use std::process::Command;
    // Format: name, memory.total, memory.free, compute_cap
    // However, memory is returned like "24576 MiB"
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits"
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() == 4 {
            let name = parts[0].to_string();
            // Convert MiB to Bytes
            let vram_total = parts[1].parse::<usize>().unwrap_or(0) * 1024 * 1024;
            let vram_available = parts[2].parse::<usize>().unwrap_or(0) * 1024 * 1024;
            let compute_capability = Some(parts[3].to_string());

            gpus.push(AcceleratorInfo {
                kind: AcceleratorType::Nvidia,
                name,
                vram_total,
                vram_available,
                compute_capability,
            });
        }
    }

    if gpus.is_empty() {
        None
    } else {
        Some(gpus)
    }
}

fn detect_amd() -> Option<Vec<AcceleratorInfo>> {
    use std::process::Command;
    // Attempt rocm-smi
    let output = Command::new("rocm-smi")
        .args(["--showid", "--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut gpus = Vec::new();
    
    // Naive parsing for rocm-smi csv output
    // Example: device,id,vram Total Memory (B),vram Total Used Memory (B)
    for line in stdout.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 4 {
            let name = format!("AMD GPU {}", parts[1]);
            let vram_total = parts[2].parse::<usize>().unwrap_or(0);
            let vram_used = parts[3].parse::<usize>().unwrap_or(0);
            let vram_available = vram_total.saturating_sub(vram_used);
            
            gpus.push(AcceleratorInfo {
                kind: AcceleratorType::Amd,
                name,
                vram_total,
                vram_available,
                compute_capability: None,
            });
        }
    }

    if gpus.is_empty() { None } else { Some(gpus) }
}

fn detect_intel() -> Option<Vec<AcceleratorInfo>> {
    use std::process::Command;
    // Attempt sycl-ls or similar for Level Zero
    let output = Command::new("sycl-ls").output().ok()?;
    
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut gpus = Vec::new();

    // Naive parsing: just check if an Intel GPU is listed
    for line in stdout.lines() {
        if line.contains("ext_oneapi_level_zero") && line.contains("GPU") {
            gpus.push(AcceleratorInfo {
                kind: AcceleratorType::Intel,
                name: "Intel oneAPI GPU".to_string(),
                vram_total: 8_000_000_000, // Hardcoded fallback for detection stub
                vram_available: 8_000_000_000,
                compute_capability: None,
            });
        }
    }

    if gpus.is_empty() { None } else { Some(gpus) }
}

fn detect_apple() -> Option<Vec<AcceleratorInfo>> {
    use std::process::Command;
    let output = Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    if stdout.contains("Apple M") || stdout.contains("Metal") {
        // Apple Unified Memory - just use sysinfo RAM
        use sysinfo::System;
        let mut sys = System::new_all();
        sys.refresh_all();
        let total = sys.total_memory() as usize;
        let available = sys.available_memory() as usize;

        return Some(vec![AcceleratorInfo {
            kind: AcceleratorType::AppleMetal,
            name: "Apple Metal GPU".to_string(),
            vram_total: total,
            vram_available: available,
            compute_capability: None,
        }]);
    }
    None
}
