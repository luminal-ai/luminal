<#
One-step helper to build and run Luminal on Windows after prerequisites are installed.
Usage: open PowerShell, cd to repo root (luminal\luminal) and run: .\scripts\run-windows.ps1

It checks for Rust/Cargo, updates cudarc if necessary, builds the CUDA crate and runs the matmul demo.
If a step fails it prints actionable next steps.
#>

Write-Host "== Luminal: run-windows helper =="

function ExitWith($code, $msg) {
    Write-Host $msg
    exit $code
}

Write-Host "Checking for Cargo..."
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    ExitWith 1 "cargo not found. Please run scripts\setup-windows.ps1 or install rustup and restart PowerShell."
}

Write-Host "cargo: $(cargo --version)"
Write-Host "rustc: $(rustc --version)"

Write-Host "Current git branch:"; git branch --show-current

Write-Host "Updating cudarc in lockfile (if applicable)..."
try {
    cargo update -p cudarc -q
    Write-Host "cargo update -p cudarc completed (or no change)."
} catch {
    Write-Host "cargo update -p cudarc failed or not needed; continuing. Error: $_"
}

Write-Host "Building luminal_cuda (release, with cuda feature). This may take a while..."
$build = cargo build -p luminal_cuda --release --features cuda 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed. Output:"
    Write-Host $build
    Write-Host "Common causes: missing Visual Studio Build Tools (C++), missing CUDA toolkit or mismatched driver/toolkit."
    Write-Host "Please ensure:\n - Visual Studio Build Tools (Desktop dev with C++) installed\n - CUDA toolkit installed and CUDA_HOME is set to the correct path\n - NVIDIA drivers are installed"
    ExitWith 2 "Build failed. See output above."
}

Write-Host "Build succeeded. Running matmul demo..."
$run = cargo run -p matmul --release --features cuda 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Demo failed. Output:"
    Write-Host $run
    ExitWith 3 "Demo failed. See output above."
}

Write-Host "Demo finished successfully. Output:"
Write-Host $run
Write-Host "Done. If you see runtime errors about missing CUDA symbols (eg. cuCtxCreate_v4), check CUDA driver/toolkit compatibility and try the steps in docs/cuda_setup.md"
