# Copies the installed NvPresent64.dll into ..\ref so it can be inspected offline.
# Read-only on the driver side: nothing in DriverStore is changed.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$ref = Join-Path $root 'ref'
New-Item -ItemType Directory -Force -Path $ref | Out-Null

$store = Join-Path $env:WINDIR 'System32\DriverStore\FileRepository'
$found = @(Get-ChildItem -Path $store -Directory -Filter 'nv*.inf_amd64_*' |
    ForEach-Object { Join-Path $_.FullName 'NvPresent64.dll' } |
    Where-Object { Test-Path $_ } |
    ForEach-Object { Get-Item $_ } |
    Sort-Object LastWriteTime -Descending)

if ($found.Count -eq 0) {
    Write-Host 'NvPresent64.dll was not found in the driver store.' -ForegroundColor Red
    Write-Host 'Smooth Motion needs a recent GeForce driver; check the driver version and try again.'
    exit 1
}

$dll = $found[0]
$target = Join-Path $ref 'NvPresent64.dll'
Copy-Item -Path $dll.FullName -Destination $target -Force

$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' } | Select-Object -First 1
$windowsVersion = if ($gpu) { $gpu.DriverVersion } else { 'unknown' }
$nvidiaVersion = 'unknown'
$digits = $windowsVersion -replace '\.', ''
if ($digits.Length -ge 5) {
    $tail = $digits.Substring($digits.Length - 5)
    $nvidiaVersion = $tail.Substring(0, 3) + '.' + $tail.Substring(3)
}

$lines = @(
    "copied_at       = $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "gpu             = $(if ($gpu) { $gpu.Name } else { 'unknown' })"
    "driver_windows  = $windowsVersion"
    "driver_nvidia   = $nvidiaVersion"
    "source          = $($dll.FullName)"
    "size_bytes      = $($dll.Length)"
    "last_write      = $($dll.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    "sha256          = $((Get-FileHash -Path $target -Algorithm SHA256).Hash.ToLower())"
    "copies_in_store = $($found.Count)"
)
foreach ($other in $found) { $lines += "store_copy      = $($other.FullName) ($($other.Length) bytes)" }
Set-Content -Path (Join-Path $ref 'driver_info.txt') -Value $lines -Encoding UTF8

Write-Host ''
Write-Host "Copied NvPresent64.dll ($([math]::Round($dll.Length / 1MB, 1)) MB) from driver $nvidiaVersion" -ForegroundColor Green
Write-Host "  to $target"
Write-Host 'Details are in ref\driver_info.txt. You can close this window.'
