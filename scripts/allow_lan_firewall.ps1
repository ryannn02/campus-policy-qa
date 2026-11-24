# 一键放行局域网访问：为项目 venv 的 python.exe 添加 TCP 8000 入站规则（仅限本地子网）。
# 需要管理员权限：右键「使用 PowerShell 运行」或在弹出的 UAC 中点「是」。
# 规则范围很窄：只放行本项目的 python.exe、只放行 8000 端口、只允许同一局域网的设备访问。

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "找不到 $python，请先在项目根目录执行 uv sync" -ForegroundColor Yellow
    exit 1
}

$port = 8000
if ($args.Count -gt 0) { $port = [int]$args[0] }
$name = "Campus QA Web (TCP $port, LocalSubnet)"

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    # 以管理员身份重新启动自己
    Start-Process -FilePath 'pwsh' -Verb RunAs -Wait -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath, "$port"
    )
    $rule = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if ($rule) { Write-Host "已添加防火墙规则：$name" -ForegroundColor Green }
    else { Write-Host "规则未添加（可能取消了 UAC 确认）" -ForegroundColor Yellow }
    exit $(if ($rule) { 0 } else { 1 })
}

Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow `
    -Program $python -Protocol TCP -LocalPort $port -RemoteAddress LocalSubnet -Profile Any | Out-Null
Write-Host "已添加防火墙规则：$name" -ForegroundColor Green
