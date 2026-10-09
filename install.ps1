param(
    [string]$Version = 'latest',
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\embeddinggemma2-jetha')
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() -ne 'X64' -or
    [Environment]::OSVersion.Platform -ne 'Win32NT') {
    throw 'This installer supports Windows x86_64 CPU; other targets require a source build.'
}
if ($Version -notmatch '^[A-Za-z0-9._-]{1,64}$') { throw 'Invalid release version.' }
$InstallDir = [IO.Path]::GetFullPath($InstallDir)
$release = if ($Version -eq 'latest') { 'latest/download' } else { "download/$Version" }
$base = "https://github.com/jethac/embeddinggemma2-jetha.c/releases/$release"
$asset = 'embeddinggemma2-jetha-windows-x86_64-cpu'
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('embeddinggemma2-jetha-' + [Guid]::NewGuid().ToString('N'))
$scratch = [IO.Path]::GetFullPath($scratch)
if (-not $scratch.StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Temporary path escaped its parent.'
}
New-Item -ItemType Directory -Path $scratch | Out-Null
try {
    $checksums = Join-Path $scratch 'SHA256SUMS'
    Invoke-WebRequest "$base/SHA256SUMS" -OutFile $checksums
    $hashes = @{}
    foreach ($name in @("$asset.exe", "$asset.runtime.zip")) {
        $entries = @(Get-Content -LiteralPath $checksums | Where-Object { $_ -match ('^([0-9a-fA-F]{64})\s+\*?' + [regex]::Escape($name) + '$') })
        if ($entries.Count -ne 1) { throw "SHA256SUMS must contain exactly one entry for $name" }
        $expected = ($entries[0] -split '\s+')[0].ToLowerInvariant()
        $file = Join-Path $scratch $name
        Invoke-WebRequest "$base/$name" -OutFile $file
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
            throw "Checksum verification failed for $name"
        }
        $hashes[$name] = $expected
    }
    $prefix = Join-Path $scratch 'app'
    New-Item -ItemType Directory -Path $prefix | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead((Join-Path $scratch "$asset.runtime.zip"))
    try {
        foreach ($entry in $archive.Entries) {
            $path = [IO.Path]::GetFullPath((Join-Path $prefix $entry.FullName))
            if (-not $path.StartsWith($prefix + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
                $entry.FullName -match ':' -or $entry.FullName -notmatch '^(bin|lib|share)/' -or
                (($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000) {
                throw "Unsafe runtime archive entry: $($entry.FullName)"
            }
        }
    } finally { $archive.Dispose() }
    Expand-Archive -LiteralPath (Join-Path $scratch "$asset.runtime.zip") -DestinationPath $prefix
    $binary = Join-Path $prefix 'bin\embeddinggemma2-jetha.exe'
    Copy-Item -LiteralPath (Join-Path $scratch "$asset.exe") -Destination $binary
    foreach ($required in @('bin\ggml-cpu-x64.dll', 'share\licenses\embeddinggemma2-jetha\LICENSE')) {
        if (-not (Test-Path -LiteralPath (Join-Path $prefix $required) -PathType Leaf)) {
            throw "Runtime archive is missing $required"
        }
    }
    # Windows PowerShell 5 turns native stderr into PowerShell errors, even for
    # successful --help output. Judge this native probe by its process exit code.
    $probePreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $binary --help *> $null
    } finally { $ErrorActionPreference = $probePreference }
    if ($LASTEXITCODE -ne 0) { throw "Staged executable failed to load: $LASTEXITCODE" }
    $revision = $Version + '-' + $hashes["$asset.exe"].Substring(0,12) + '-' + $hashes["$asset.runtime.zip"].Substring(0,12)
    $versions = Join-Path $InstallDir 'versions'
    $destination = Join-Path $versions $revision
    New-Item -ItemType Directory -Force -Path $versions | Out-Null
    if (Test-Path -LiteralPath $destination) {
        $existing = Join-Path $destination 'bin\embeddinggemma2-jetha.exe'
        if (-not (Test-Path -LiteralPath $existing) -or
            (Get-FileHash -LiteralPath $existing -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashes["$asset.exe"]) {
            throw "Existing installation differs; choose a new InstallDir: $destination"
        }
    } else {
        Move-Item -LiteralPath $prefix -Destination $destination
    }
    $launcher = Join-Path $InstallDir 'embeddinggemma2-jetha.cmd'
    $temporaryLauncher = Join-Path $InstallDir ('.launcher-' + [Guid]::NewGuid().ToString('N') + '.cmd')
    $content = "@echo off`r`n`"%~dp0versions\$revision\bin\embeddinggemma2-jetha.exe`" %*`r`nexit /b %errorlevel%`r`n"
    [IO.File]::WriteAllText($temporaryLauncher, $content, [Text.Encoding]::ASCII)
    Move-Item -LiteralPath $temporaryLauncher -Destination $launcher -Force
    Write-Output "Installed $Version. Run $launcher; add $InstallDir to PATH for direct use."
    Write-Output 'Model weights are separate. See the README for model download and multimodal startup; video/WebP require FFmpeg on PATH.'
} finally {
    # This is the exact temporary directory created above, never the install root.
    if (-not [IO.Path]::GetFullPath($scratch).StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()), [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing cleanup outside the temporary directory.'
    }
    Remove-Item -LiteralPath $scratch -Recurse -Force
}
