@echo off
chcp 65001 >nul
echo ============================================
echo  Патч DPVO для PyTorch 2.10+
echo ============================================
echo.

if "%1"=="" (
    echo Использование: patch_dpvo.bat путь\к\DPVO
    echo Пример:        patch_dpvo.bat C:\Users\user\DPVO
    exit /b 1
)

set "DPVO_DIR=%~1"

if not exist "%DPVO_DIR%\setup.py" (
    echo ОШИБКА: setup.py не найден в %DPVO_DIR%
    echo Убедитесь что указан корень репозитория DPVO
    exit /b 1
)

echo Каталог DPVO: %DPVO_DIR%
echo.

:: --- correlation_kernel.cu ---
set "FILE=%DPVO_DIR%\dpvo\altcorr\correlation_kernel.cu"
if exist "%FILE%" (
    echo Патчим: %FILE%
    powershell -Command "(Get-Content '%FILE%') -replace 'fmap1\.type\(\)', 'fmap1.scalar_type()' -replace 'net\.type\(\)', 'net.scalar_type()' | Set-Content '%FILE%'"
    echo   OK
) else (
    echo ПРОПУСК: %FILE% не найден
)

:: --- lietorch_gpu.cu ---
set "FILE=%DPVO_DIR%\dpvo\lietorch\src\lietorch_gpu.cu"
if exist "%FILE%" (
    echo Патчим: %FILE%
    powershell -Command "(Get-Content '%FILE%') -replace '(\b[aX])\.type\(\)', '$1.scalar_type()' | Set-Content '%FILE%'"
    echo   OK
) else (
    echo ПРОПУСК: %FILE% не найден
)

:: --- lietorch_cpu.cpp ---
set "FILE=%DPVO_DIR%\dpvo\lietorch\src\lietorch_cpu.cpp"
if exist "%FILE%" (
    echo Патчим: %FILE%
    powershell -Command "(Get-Content '%FILE%') -replace '(\b[aX])\.type\(\)', '$1.scalar_type()' | Set-Content '%FILE%'"
    echo   OK
) else (
    echo ПРОПУСК: %FILE% не найден
)

:: --- dispatch.h ---
set "FILE=%DPVO_DIR%\dpvo\lietorch\include\dispatch.h"
if exist "%FILE%" (
    echo Патчим: %FILE%
    powershell -Command ^
        "$c = Get-Content '%FILE%' -Raw; " ^
        "$old = '    const auto& the_type = TYPE;                                                     \' + \"`n\" + '    /* don''t use TYPE again in case it is an expensive or side-effect op */          \' + \"`n\" + '    at::ScalarType _st = ::detail::scalar_type(the_type);                            \'; " ^
        "$new = '    at::ScalarType _st = TYPE;                                                       \'; " ^
        "$c = $c -replace [regex]::Escape($old), $new; " ^
        "Set-Content '%FILE%' $c"
    echo   OK
) else (
    echo ПРОПУСК: %FILE% не найден
)

echo.
echo ============================================
echo  Патч завершён
echo ============================================
echo.
echo Теперь соберите DPVO из Developer Command Prompt for VS 2022:
echo   cd %DPVO_DIR%
echo   set DISTUTILS_USE_SDK=1
echo   pip install . --no-build-isolation
echo.
