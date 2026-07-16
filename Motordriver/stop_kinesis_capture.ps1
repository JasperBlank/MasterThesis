$ErrorActionPreference = "Stop"

$outDir = Join-Path $PSScriptRoot "captures"
$etl = Join-Path $outDir "kinesis_capture.etl"
$pcap = Join-Path $outDir "kinesis_capture.pcapng"

if (!(Test-Path $outDir)) {
    throw "Capture directory does not exist. Run start_kinesis_capture.ps1 first."
}

$stopOutput = pktmon stop 2>&1
if ($LASTEXITCODE -ne 0) {
    $stopOutput | Write-Host
    throw "pktmon was not running. Run start_kinesis_capture.ps1 from an Administrator PowerShell first."
}

if (!(Test-Path $etl)) {
    throw "Missing $etl. Capture did not start correctly."
}

if (Test-Path $pcap) {
    Remove-Item -LiteralPath $pcap -Force
}
$convertOutput = pktmon etl2pcap $etl --out $pcap 2>&1
if ($LASTEXITCODE -ne 0 -or !(Test-Path $pcap)) {
    $convertOutput | Write-Host
    throw "Could not convert capture to pcapng."
}

Write-Host "Wrote $pcap"
Write-Host "Run: python parse_kinesis_capture.py"
