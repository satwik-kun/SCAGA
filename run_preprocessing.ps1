# run_preprocessing.ps1
# Runs preprocessing on all subset0 scans, one at a time (avoids TIGRE slowdown bug).
# Skips scans already processed. Automatically resolves paths relative to repository root.

$PROJECT_ROOT = $PSScriptRoot
$PREPROCESS_DIR = "$PROJECT_ROOT\Preprocessing\data\LUNA16"
$PYTHONPATH_DIR = "$PROJECT_ROOT\Preprocessing\data"
$NAMES_FILE     = "$PREPROCESS_DIR\raw\subset0_names.txt"
$PROCESSED_DIR  = "$PREPROCESS_DIR\processed\images"

if (-not (Test-Path $NAMES_FILE)) {
    Write-Host "[ERROR] Could not find scan names list at: $NAMES_FILE" -ForegroundColor Red
    Write-Host "Please ensure your dataset is placed according to machine_config.yaml." -ForegroundColor Yellow
    exit 1
}

$names = Get-Content $NAMES_FILE | Where-Object { $_ -ne "" }
$total  = $names.Count
$done   = 0
$skipped = 0

Write-Host "=========================================="
Write-Host "Preprocessing $total subset0 scans"
Write-Host "=========================================="

foreach ($name in $names) {
    $out_file = "$PROCESSED_DIR\$name.nii.gz"

    if (Test-Path $out_file) {
        $skipped++
        Write-Host "[SKIP $($done+$skipped)/$total] $name"
        continue
    }

    Write-Host "[PROC $($done+1+$skipped)/$total] $name ..."

    $env:PYTHONPATH = $PYTHONPATH_DIR
    $proc = Start-Process -FilePath "py" `
        -ArgumentList "-3.14", "main.py", "-n", $name `
        -WorkingDirectory $PREPROCESS_DIR `
        -PassThru -Wait -NoNewWindow

    if ($proc.ExitCode -eq 0) {
        $done++
        Write-Host "  -> OK ($done done)"
    } else {
        Write-Host "  -> FAILED (exit code $($proc.ExitCode))"
    }
}

Write-Host "=========================================="
Write-Host "Done: $done processed, $skipped skipped"
Write-Host "=========================================="
