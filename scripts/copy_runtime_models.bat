@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Copy the deployed model set into a dist models folder:
rem   gguf\qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf   v3.9 multitask generator (all tasks)
rem   qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged\   FedProx Bloom classifier
rem   bge-small\                                           BAAI/bge-small-en-v1.5 retrieval encoder
rem Usage: scripts\copy_runtime_models.bat <dest_models_dir>

set "DEST=%~1"
if "%DEST%"=="" (
    echo ERROR: destination models directory required.
    exit /b 1
)

set "GEN=models\gguf\qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf"
set "BLOOM_NAME=qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged"
set "BLOOM_SRC=models\%BLOOM_NAME%"

if not exist "%GEN%" (
    echo ERROR: required model file missing: %GEN%
    echo Create it with experiments\multitask_bloom_rewrite\scripts\convert_to_gguf.py
    exit /b 1
)
if not exist "%BLOOM_SRC%\model.safetensors" (
    echo ERROR: FedProx Bloom classifier weights missing: %BLOOM_SRC%\model.safetensors
    exit /b 1
)

rem BGE: prefer models\bge-small, otherwise the Hugging Face cache snapshot of BAAI/bge-small-en-v1.5.
set "BGE_SRC="
if exist "models\bge-small\model.safetensors" set "BGE_SRC=models\bge-small"
if not defined BGE_SRC (
    for /d %%S in ("%USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5\snapshots\*") do (
        if exist "%%~S\model.safetensors" set "BGE_SRC=%%~S"
    )
)
if not defined BGE_SRC (
    echo ERROR: BGE encoder not found in models\bge-small or the Hugging Face cache.
    echo Download it once with:
    echo   .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5')"
    exit /b 1
)

if exist "%DEST%" rmdir /s /q "%DEST%"
mkdir "%DEST%\gguf" || exit /b 1

echo Copying v3.9 generator GGUF...
copy /Y "%GEN%" "%DEST%\gguf\" >nul || exit /b 1

echo Writing FedProx r20 Bloom classifier (fp16 storage, loaded as fp32 on CPU; halves 1.9 GB)...
rem Same predictions as the fp32 file on all 349 validation questions; see scripts\export_classifier_fp16.py.
set "PY=.venv-build\Scripts\python.exe"
if not exist "%PY%" set "PY=.venv\Scripts\python.exe"
"%PY%" scripts\export_classifier_fp16.py "%BLOOM_SRC%" "%DEST%\%BLOOM_NAME%"
if errorlevel 1 (
    echo ERROR: classifier export failed.
    exit /b 1
)

echo Copying BGE encoder from !BGE_SRC! ...
rem /SL is not used: HF cache files are copied as real files. ONNX and .bin duplicates are skipped.
robocopy "!BGE_SRC!" "%DEST%\bge-small" /E /XD .cache onnx /XF pytorch_model.bin *.onnx README.md .gitattributes *.metadata /NFL /NDL /NJH /NJS /NC /NS /NP
if errorlevel 8 exit /b 1

for %%F in ("gguf\qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf" "%BLOOM_NAME%\model.safetensors" "bge-small\model.safetensors" "bge-small\tokenizer.json") do (
    if not exist "%DEST%\%%~F" (
        echo ERROR: copy verification failed: %DEST%\%%~F
        exit /b 1
    )
)

(
    echo {
    echo   "generator": {"id": "multitask_eduguard_v3_9_q4_k_m", "path": "models/gguf/qwen15b_multitask_eduguard_v3_9_q4_k_m.gguf"},
    echo   "bloom_classifier": {"id": "fedprox_iid_r20_best_r20_merged", "base": "Qwen2.5-0.5B-Instruct", "storage": "fp16, loaded as fp32 on CPU", "path": "models/%BLOOM_NAME%"},
    echo   "retrieval_encoder": {"id": "BAAI/bge-small-en-v1.5", "path": "models/bge-small"}
    echo }
) > "%DEST%\deployment_manifest.json"

echo Runtime models ready at %DEST%
exit /b 0
