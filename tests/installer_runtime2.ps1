param([Parameter(Mandatory)][string]$AssetsDir)
$ErrorActionPreference = 'Stop'
$probeAssets = (Resolve-Path -LiteralPath $AssetsDir).Path
$probeRoot = Join-Path ([IO.Path]::GetTempPath()) ('installer-runtime2-' + [Guid]::NewGuid().ToString('N'))
$probeRoot = [IO.Path]::GetFullPath($probeRoot)
New-Item -ItemType Directory -Path $probeRoot | Out-Null
$probeCorrupt = $false
function Test-Launcher([string]$Launcher) {
    $ErrorActionPreference = 'Continue'
    & $Launcher --help *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Installed command cannot load its runtime' }
}
function Invoke-WebRequest {
    param([string]$Uri, [string]$OutFile)
    if (-not $Uri.StartsWith('https://github.com/jethac/embeddinggemma2-jetha.c/releases/latest/download/')) { throw "Unexpected URL: $Uri" }
    $name = [IO.Path]::GetFileName(([Uri]$Uri).AbsolutePath)
    if ($name -eq 'SHA256SUMS') {
        $entries = Get-ChildItem -LiteralPath $probeAssets -File | ForEach-Object {
            (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $_.Name
        }
        [IO.File]::WriteAllLines($OutFile, [string[]]$entries)
    } else {
        Copy-Item -LiteralPath (Join-Path $probeAssets $name) -Destination $OutFile
        if ($probeCorrupt -and $name.EndsWith('.runtime.zip')) { [IO.File]::AppendAllText($OutFile, 'corrupted download') }
    }
}
try {
    $installer = Join-Path $PSScriptRoot '..\install.ps1'
    $install = Join-Path $probeRoot 'installed app with spaces'
    & $installer -InstallDir $install
    $launcher = Join-Path $install 'embeddinggemma2-jetha.cmd'
    Test-Launcher $launcher
    $before = (Get-FileHash -LiteralPath $launcher -Algorithm SHA256).Hash
    $probeCorrupt = $true
    $rejected = $false
    try { & $installer -InstallDir $install } catch {
        if ($_.Exception.Message -notlike '*Checksum verification failed*') { throw }
        $rejected = $true
    }
    if (-not $rejected -or (Get-FileHash -LiteralPath $launcher -Algorithm SHA256).Hash -ne $before) {
        throw 'Corrupt download replaced the working installation'
    }
    Test-Launcher $launcher
    Write-Output 'Windows runtime installer loads with spaces and preserves the working installation after checksum failure: passed'
} finally {
    if (-not [IO.Path]::GetFullPath($probeRoot).StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()), [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing cleanup outside temporary directory'
    }
    Remove-Item -LiteralPath $probeRoot -Recurse -Force
}
