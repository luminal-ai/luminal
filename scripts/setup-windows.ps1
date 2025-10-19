<#
Windows helper script to bootstrap Rust and provide instructions for native build tools and CUDA.
This script installs rustup (non-interactive), updates PATH for the session, and prints next steps.
It cannot install Visual Studio Build Tools or CUDA automatically; please follow the links shown.
#>

Write-Host "== Luminal: Windows bootstrap helper =="

Write-Host "1) Installing rustup (stable toolchain)..."
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
Invoke-WebRequest -Uri https://sh.rustup.rs -UseBasicParsing | Invoke-Expression

Write-Host "Adding Cargo to PATH for this session..."
$env:PATH = $env:PATH + ";$env:USERPROFILE\.cargo\bin"

Write-Host "Rust version:"; & cargo --version
Write-Host "Rustc version:"; & rustc --version

Write-Host "\n2) Please install Visual Studio Build Tools (C++ workload) manually if not already installed."
Write-Host "   Download: https://visualstudio.microsoft.com/downloads/ -> 'Build Tools for Visual Studio'"
Write-Host "   During install select: 'Desktop development with C++' (MSVC, Windows SDK, CMake)"

Write-Host "\n3) Install NVIDIA driver and CUDA toolkit (recommended CUDA 12.2 for recent changes)."
Write-Host "   CUDA downloads: https://developer.nvidia.com/cuda-downloads"

Write-Host "\n4) After installing the above, open a new PowerShell and run these commands to build the project:";
Write-Host "   cd \"$(Resolve-Path ..)\"";
Write-Host "   git fetch origin";
Write-Host "   git checkout main";
Write-Host "   cargo update -p cudarc";
Write-Host "   cargo build -p luminal_cuda --release --features cuda";

Write-Host "\nIf you hit issues, copy the full 'cargo build' output and paste it into the issue or here so I can help iterate.";
