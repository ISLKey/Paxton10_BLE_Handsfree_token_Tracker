@echo off
title BLE Interactive Floor Plan System v3.3.1 - JavaScript Fixes
echo ================================================================
echo  🔧 BLE INTERACTIVE FLOOR PLAN SYSTEM v3.3.1
echo  ✅ JAVASCRIPT ERRORS FIXED - TABS NOW WORKING
echo ================================================================

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
echo 🔧 v3.3.1 FIXES:
echo - ✅ Fixed duplicate 'isDragging' variable declaration
echo - ✅ Fixed duplicate 'dragOffset' variable declaration  
echo - ✅ Renamed floor plan variables to avoid conflicts
echo - ✅ All tabs now clickable and functional
echo - ✅ Drag and drop still works in both canvases
echo.
echo 🎯 FEATURES:
echo - ✅ Drag and drop furniture items in Interactive Floor Plan
echo - ✅ Select tool for object manipulation and repositioning
echo - ✅ Visual feedback during dragging operations
echo - ✅ Console logging for debugging drag operations
echo - ✅ Working tab navigation (Overview, Visual Mapping, etc.)
echo.
echo 🎮 HOW TO USE:
echo 1. All tabs should now be clickable
echo 2. Go to Interactive Floor Plan tab
echo 3. Click furniture items to add them to the canvas
echo 4. Use Select tool (cursor icon) to drag items around
echo 5. Watch console for drag operation feedback
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

