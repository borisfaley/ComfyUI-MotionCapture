@echo off
echo ============================================
echo  Patch DPVO for PyTorch 2.10+
echo ============================================
echo.

if "%1"=="" (
    echo Usage: patch_dpvo.bat path\to\DPVO
    echo Example: patch_dpvo.bat C:\Users\user\DPVO
    exit /b 1
)

set "DPVO_DIR=%~1"

if not exist "%DPVO_DIR%\setup.py" (
    echo ERROR: setup.py not found in %DPVO_DIR%
    echo Make sure you point to the DPVO repository root
    exit /b 1
)

echo DPVO dir: %DPVO_DIR%
echo.

:: --- correlation_kernel.cu ---
set "FILE=%DPVO_DIR%\dpvo\altcorr\correlation_kernel.cu"
if exist "%FILE%" (
    echo Patching: %FILE%
    powershell -Command "(Get-Content '%FILE%') -replace 'fmap1\.type\(\)', 'fmap1.scalar_type()' -replace 'net\.type\(\)', 'net.scalar_type()' | Set-Content '%FILE%'"
    echo   OK
) else (
    echo SKIP: %FILE% not found
)

:: --- lietorch_gpu.cu ---
set "FILE=%DPVO_DIR%\dpvo\lietorch\src\lietorch_gpu.cu"
if exist "%FILE%" (
    echo Patching: %FILE%
    powershell -Command "(Get-Content '%FILE%') -replace '(\b[aX])\.type\(\)', '$1.scalar_type()' | Set-Content '%FILE%'"
    echo   OK
) else (
    echo SKIP: %FILE% not found
)

:: --- lietorch_cpu.cpp ---
set "FILE=%DPVO_DIR%\dpvo\lietorch\src\lietorch_cpu.cpp"
if exist "%FILE%" (
    echo Patching: %FILE%
    powershell -Command "(Get-Content '%FILE%') -replace '(\b[aX])\.type\(\)', '$1.scalar_type()' | Set-Content '%FILE%'"
    echo   OK
) else (
    echo SKIP: %FILE% not found
)

:: --- dispatch.h ---
set "FILE=%DPVO_DIR%\dpvo\lietorch\include\dispatch.h"
if exist "%FILE%" (
    echo Patching: %FILE%
    powershell -Command ^
        "$c = Get-Content '%FILE%' -Raw; " ^
        "$old = '    const auto& the_type = TYPE;                                                     \' + \"`n\" + '    /* don''t use TYPE again in case it is an expensive or side-effect op */          \' + \"`n\" + '    at::ScalarType _st = ::detail::scalar_type(the_type);                            \'; " ^
        "$new = '    at::ScalarType _st = TYPE;                                                       \'; " ^
        "$c = $c -replace [regex]::Escape($old), $new; " ^
        "Set-Content '%FILE%' $c"
    echo   OK
) else (
    echo SKIP: %FILE% not found
)

echo.
echo ============================================
echo  Patch complete
echo ============================================
echo.
echo Now build DPVO from Developer Command Prompt for VS 2022:
echo   cd %DPVO_DIR%
echo   set DISTUTILS_USE_SDK=1
echo   pip install . --no-build-isolation
echo.
