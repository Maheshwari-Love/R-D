@echo off
echo ============================================================
echo   Change-Agent Environment Setup
echo ============================================================
echo.

:: Step 1: Create and activate conda env
echo [1/7] Creating conda environment: change_agent_env (Python 3.9)
call conda create -n change_agent_env python=3.9 -y
call conda activate change_agent_env

:: Step 2: Fix setuptools version (must be <70 for torch 2.0.1 compatibility)
echo.
echo [2/7] Installing compatible setuptools...
pip install "setuptools<70" -q

:: Step 3: Detect GPU
echo.
echo [3/7] Detecting GPU...
nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo      GPU detected! Installing PyTorch with CUDA 11.8...
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
) ELSE (
    echo      No GPU detected. Installing PyTorch CPU-only...
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
)

:: Step 4: Install mmcv-full (pre-built wheel)
echo.
echo [4/7] Installing mmcv-full 1.7.2...
nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
) ELSE (
    pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
)

:: Step 5: Install mmsegmentation (compatible with mmcv 1.7.2)
echo.
echo [5/7] Installing mmsegmentation 0.30.0...
pip install mmsegmentation==0.30.0

:: Step 6: Install remaining core dependencies
echo.
echo [6/7] Installing remaining dependencies...
pip install numpy==1.25.2 openai==1.3.4 opencv-python==4.8.0.74 pandas==2.1.2 Pillow==10.0.1 pyarrow==14.0.0 tqdm==4.66.4 transformers==4.33.1 mmengine==0.9.1 scikit-image imageio einops

:: Step 7: Install lagent agent framework
echo.
echo [7/7] Installing lagent agent framework...
cd lagent-main
pip install -e . -q
pip install streamlit tiktoken google-search-results func_timeout -q
cd ..

echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Download MCI_model.pth from:
echo      https://huggingface.co/lcybuaa/Change-Agent/tree/main
echo      Place it in: Multi_change\models_ckpt\MCI_model.pth
echo.
echo   2. Run inference:
echo      cd Multi_change
echo      python predict.py --imgA_path "path\to\A.png" --imgB_path "path\to\B.png" --mask_save_path ".\CDmask.png"
echo.
pause
