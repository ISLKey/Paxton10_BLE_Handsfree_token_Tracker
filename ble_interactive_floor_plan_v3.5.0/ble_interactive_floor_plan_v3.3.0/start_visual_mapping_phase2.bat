@echo off
echo ========================================
echo BLE Visual Mapping System - Phase 2
echo Real-Time Location Tracking
echo ========================================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as Administrator - Good!
) else (
    echo This script requires Administrator privileges.
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Change to the directory where this batch file is located
cd /d "%~dp0"
echo Current directory: %CD%

echo.
echo Installing Python packages...
pip install flask flask-cors paho-mqtt

echo.
echo Starting BLE Visual Mapping System - Phase 2...
echo.
echo Features:
echo - Real-time device positioning on interactive floor plans
echo - ESP32 node positioning with drag-and-drop
echo - Color-coded BLE devices based on distance
echo - Movement trails showing device paths
echo - Predictive path projection
echo - Layer controls for different data types
echo.

REM Check if the Python file exists
if exist "ble_visual_mapping_phase2_fixed.py" (
    echo Found Python file, starting application...
    python ble_visual_mapping_phase2_fixed.py
) else (
    echo ERROR: ble_visual_mapping_phase2_fixed.py not found in current directory!
    echo Current directory: %CD%
    echo Files in directory:
    dir *.py
    echo.
    echo Please make sure you're running this batch file from the correct folder.
)

echo.
echo System stopped. Press any key to exit...
pause

