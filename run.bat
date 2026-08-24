@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   TIMDR-Earthquake-Core - uruchamianie GUI
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [BLAD] Nie znaleziono Pythona w PATH.
        echo Zainstaluj Python 3.10+ z https://www.python.org/downloads/
        echo Podczas instalacji zaznacz "Add python.exe to PATH".
        echo.
        pause
        exit /b 1
    ) else (
        set "PYCMD=py"
    )
) else (
    set "PYCMD=python"
)

echo Uzywam: %PYCMD%
%PYCMD% --version

echo.
echo Sprawdzanie modulu tkinter...
%PYCMD% -c "import tkinter" 2>nul
if errorlevel 1 (
    echo [BLAD] Twoja instalacja Pythona nie ma modulu tkinter.
    echo Zainstaluj ponownie Python ze strony python.org (tkinter jest
    echo dolaczony domyslnie w standardowym instalatorze Windows^).
    echo.
    pause
    exit /b 1
)

echo.
echo Instalacja/aktualizacja zaleznosci (numpy, scipy, matplotlib)...
%PYCMD% -m pip install --quiet --disable-pip-version-check numpy scipy matplotlib
if errorlevel 1 (
    echo [OSTRZEZENIE] Instalacja zaleznosci nie w pelni sie powiodla.
    echo Sprobuje mimo to uruchomic aplikacje...
)

echo.
echo Uruchamianie GUI...
echo.
%PYCMD% gui_app.py

if errorlevel 1 (
    echo.
    echo [BLAD] Aplikacja zakonczyla sie bledem - zobacz komunikat wyzej.
    pause
)

endlocal
