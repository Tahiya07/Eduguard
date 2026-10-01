@echo off
setlocal EnableExtensions

rem Copy the FINAL deployed pair into a dist models folder:
rem   1) FedProx IID r20 best merged Bloom classifier (Qwen2.5-0.5B)
rem   2) Multitask v3 Q4_K_M generator GGUF
rem   +) lean BGE-small for retrieval only
rem Usage: scripts\copy_runtime_models.bat <dest_models_dir>

set "DEST=%~1"
if "%DEST%"=="" (
    echo ERROR: destination models directory required.
    exit /b 1
)

set "GGUF=models\qwen15b_multitask_v3_q4_k_m.gguf"
set "BLOOM_SRC=models\qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged"

if not exist "%BLOOM_SRC%\model.safetensors" (
    echo ERROR: Final FedProx r20 Bloom weights are missing.
    echo Expected model.safetensors in: %BLOOM_SRC%
    echo.
    echo The tokenizer-only folder is not enough. Place the merged FedProx
    echo checkpoint weights there, then re-run.
    exit /b 1
)

if not exist "%GGUF%" (
    echo ERROR: Final multitask v3 GGUF missing: %GGUF%
    echo Place qwen15b_multitask_v3_q4_k_m.gguf in models.
    exit /b 1
)

if not exist "models\bge-small\model.safetensors" (
    echo ERROR: models\bge-small\model.safetensors missing.
    exit /b 1
)

if exist "%DEST%" rmdir /s /q "%DEST%"
mkdir "%DEST%"
mkdir "%DEST%\qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged"
mkdir "%DEST%\bge-small"

echo Copying multitask v3 generator GGUF...
copy /Y "%GGUF%" "%DEST%\qwen15b_multitask_v3_q4_k_m.gguf" >nul
if errorlevel 1 exit /b 1

echo Copying FedProx r20 Bloom classifier from %BLOOM_SRC%...
robocopy "%BLOOM_SRC%" "%DEST%\qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged" /E /XF *.zip /NFL /NDL /NJH /NJS /nc /ns /np
if errorlevel 8 exit /b 1

if not exist "%DEST%\qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged\model.safetensors" (
    echo ERROR: FedProx weights did not copy.
    exit /b 1
)

echo Copying lean BGE encoder ^(safetensors only^)...
robocopy "models\bge-small" "%DEST%\bge-small" /E /XD .cache onnx /XF pytorch_model.bin README.md .gitattributes *.metadata /NFL /NDL /NJH /NJS /nc /ns /np
if errorlevel 8 exit /b 1

(
    echo {
    echo   "bloom_classifier": {
    echo     "id": "fedprox_iid_r20_best_r20_merged",
    echo     "base": "Qwen2.5-0.5B-Instruct",
    echo     "path": "models/qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged",
    echo     "source": "%BLOOM_SRC:\=/%"
    echo   },
    echo   "generator": {
    echo     "id": "multitask_v3_q4_k_m",
    echo     "base": "Qwen2.5-1.5B-Instruct",
    echo     "path": "models/qwen15b_multitask_v3_q4_k_m.gguf",
    echo     "notes": "Final multitask Bloom rewrite / QA / summarization GGUF"
    echo   },
    echo   "retrieval_encoder": {
    echo     "id": "bge-small",
    echo     "path": "models/bge-small",
    echo     "lazy_load": true
    echo   }
    echo }
) > "%DEST%\deployment_manifest.json"

echo Final deploy models ready at %DEST%
echo   Bloom: FedProx r20 -^> qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged
echo   GGUF:  multitask v3 -^> qwen15b_multitask_v3_q4_k_m.gguf
exit /b 0
