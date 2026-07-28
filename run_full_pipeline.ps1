# run_full_pipeline.ps1
# Automates the entire Edge-Guided CT Reconstruction pipeline end-to-end for subset0.
# Completely portable across all machines and teammates using $PSScriptRoot.

$PROJECT_ROOT = $PSScriptRoot
$PREPROCESS_DIR = "$PROJECT_ROOT\Preprocessing\data\LUNA16"
$PYTHONPATH_DIR = "$PROJECT_ROOT\Preprocessing\data"
$TOTALSEG_DIR = "$PROJECT_ROOT\TotalSegmentator"
$CODE_DIR = "$PROJECT_ROOT\SCAGA"
$GEN_DIR = "$PROJECT_ROOT\generated"

Write-Host "=========================================================="
Write-Host "STARTING FULL PIPELINE AUTOMATION FOR SUBSET0"
Write-Host "=========================================================="
Write-Host "Repository Root: $PROJECT_ROOT"

# ----------------------------------------------------------
# Stage 0: Repository Health Check
# ----------------------------------------------------------
Write-Host "`n[STAGE 0/7] Running Repository Health Check..."
Set-Location $PROJECT_ROOT
py -3.14 repo_health_check.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ABORTED] Repository Health Check failed. Please review the guidance above." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------
# Stage 1: Preprocessing
# ----------------------------------------------------------
Write-Host "`n[STAGE 1/7] Ensuring all scans are preprocessed..."
$env:PYTHONPATH = $PYTHONPATH_DIR
Set-Location $PROJECT_ROOT
powershell -ExecutionPolicy Bypass -File "$PROJECT_ROOT\run_preprocessing.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ----------------------------------------------------------
# Stage 2: TotalSegmentator Edge Extraction
# ----------------------------------------------------------
Write-Host "`n[STAGE 2/7] Extracting anatomical boundaries via TotalSegmentator..."
$env:PYTHONPATH = $TOTALSEG_DIR
Set-Location $TOTALSEG_DIR
py -3.14 extract_edges.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ----------------------------------------------------------
# Stage 3: Create Probability Atlas & Importance Scores
# ----------------------------------------------------------
Write-Host "`n[STAGE 3/7] Generating edge probability atlas & importance scores..."
Set-Location $TOTALSEG_DIR
py -3.14 create_atlas.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path "$TOTALSEG_DIR\Importance_Score\compute_structural_complexity.py") {
    py -3.14 "$TOTALSEG_DIR\Importance_Score\compute_structural_complexity.py"
}
if (Test-Path "$TOTALSEG_DIR\Importance_Score\compute_importance_score.py") {
    py -3.14 "$TOTALSEG_DIR\Importance_Score\compute_importance_score.py"
}

# ----------------------------------------------------------
# Stage 4: Sample Points (SCAGA / Weighted Sum)
# ----------------------------------------------------------
Write-Host "`n[STAGE 4/7] Sampling data-driven Gaussian points..."
Set-Location $TOTALSEG_DIR
if (Test-Path "$TOTALSEG_DIR\Importance_Score\scaga_sampling.py") {
    py -3.14 "$TOTALSEG_DIR\Importance_Score\scaga_sampling.py"
} else {
    py -3.14 edge_guided_sampling.py
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ----------------------------------------------------------
# Stage 5: Ensure Artifact Integration
# ----------------------------------------------------------
Write-Host "`n[STAGE 5/7] Verifying generated artifacts in $GEN_DIR..."
# With strict artifact resolution, active development reads directly from generated/
if (-not (Test-Path "$GEN_DIR\points\sampled_points_weighted_sum.npy")) {
    if (Test-Path "$TOTALSEG_DIR\sampled_points_weighted_sum.npy") {
        New-Item -ItemType Directory -Force -Path "$GEN_DIR\points" | Out-Null
        Copy-Item -Path "$TOTALSEG_DIR\sampled_points_weighted_sum.npy" -Destination "$GEN_DIR\points\sampled_points_weighted_sum.npy" -Force
    }
}
if (-not (Test-Path "$GEN_DIR\atlases\importance_score_weighted_sum.nii.gz")) {
    if (Test-Path "$TOTALSEG_DIR\importance_score_weighted_sum.nii.gz") {
        New-Item -ItemType Directory -Force -Path "$GEN_DIR\atlases" | Out-Null
        Copy-Item -Path "$TOTALSEG_DIR\importance_score_weighted_sum.nii.gz" -Destination "$GEN_DIR\atlases\importance_score_weighted_sum.nii.gz" -Force
    }
}

# ----------------------------------------------------------
# Stage 6: Train 400 Epochs
# ----------------------------------------------------------
Write-Host "`n[STAGE 6/7] Training model for 400 epochs..."
$env:PYTHONPATH = $CODE_DIR
Set-Location $CODE_DIR
py -3.14 train.py --name SCAGA-Development --batch_size 1 --epoch 400 --dst_name LUNA16 --num_views 6 --random_views --cfg_path configs/experiment.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ----------------------------------------------------------
# Stage 7: Evaluation
# ----------------------------------------------------------
Write-Host "`n[STAGE 7/7] Evaluating trained model..."
$env:PYTHONPATH = $CODE_DIR
Set-Location $CODE_DIR
py -3.14 evaluate.py --name SCAGA-Development --epoch 400 --dst_name LUNA16 --split test --num_views 6 --out_res_scale 1.0 --save_results --cfg_path configs/experiment.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=========================================================="
Write-Host "PIPELINE COMPLETED SUCCESSFULLY!"
Write-Host "=========================================================="
