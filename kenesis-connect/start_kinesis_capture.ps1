$ErrorActionPreference = "Stop"

$hubIp = "192.168.0.200"
$outDir = Join-Path $PSScriptRoot "captures"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

cmd /c "pktmon stop >nul 2>nul"
cmd /c "pktmon filter remove >nul 2>nul"
pktmon filter add KinesisHub -i $hubIp -t TCP | Out-Null

$etl = Join-Path $outDir "kinesis_capture.etl"
if (Test-Path $etl) {
    Remove-Item -LiteralPath $etl -Force
}

$startOutput = pktmon start --capture --comp nics --pkt-size 0 --file-name $etl 2>&1
if ($LASTEXITCODE -ne 0) {
    $startOutput | Write-Host
    throw "pktmon did not start. Open PowerShell as Administrator and run this script again."
}

Write-Host "Capturing Kinesis Ethernet traffic for $hubIp"
Write-Host "Now open Kinesis, connect the hub/controllers, wait until panels update, then run stop_kinesis_capture.ps1"
