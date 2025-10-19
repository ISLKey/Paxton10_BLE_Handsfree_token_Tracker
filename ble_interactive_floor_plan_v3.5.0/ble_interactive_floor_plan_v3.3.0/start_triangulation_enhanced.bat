@echo off
echo ========================================
echo BLE Visual Mapping - MAX DISTANCE FEATURE
echo ESP32 Node Range Control + Visual Coverage
echo ========================================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as Administrator - Good!
) else (
    echo ERROR: This script must be run as Administrator
    echo Right-click this file and select "Run as administrator"
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
echo NEW MAX DISTANCE FEATURES:
echo - Set max detection distance per ESP32 node (default: 16m)
echo - Visual coverage circles on mapping (scaled to meters)
echo - MQTT integration: sends espresense/rooms/[node]/max_distance/set
echo - Edit nodes to change detection range (0 = no limit)
echo - Coverage areas show actual detection radius
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

