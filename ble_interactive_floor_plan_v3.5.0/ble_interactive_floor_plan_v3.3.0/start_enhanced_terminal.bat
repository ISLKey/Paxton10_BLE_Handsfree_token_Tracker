@echo off
echo ========================================
echo BLE Enhanced Terminal Visual Mapping System v11.0.0 - RICH TERMINAL LOGGING
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.13 and try again
    pause
    exit /b 1
)

echo Installing Python 3.13 compatible packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install required packages
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting BLE Enhanced Terminal Visual Mapping System...
echo ========================================
echo.
echo ENHANCED TERMINAL LOGGING FEATURES:
echo - Color-coded Distance Indicators (Green/Yellow/Orange/Red circles)
echo - BLE Device Names and Friendly Identifiers
echo - Real-time Position Tracking with Visual Feedback
echo - No Duplicate Node Creation - Fixed Logic
echo - DHCP Auto-detection and Network Scanning
echo - Rich Console Experience with Visual Symbols
echo.
echo Web interface will be available at:
echo - Local: http://localhost:5000
echo - Network: http://[your-ip]:5000
echo.
echo WATCH THIS TERMINAL FOR RICH, COLOR-CODED REAL-TIME LOGGING!
echo.
echo Press Ctrl+C to stop the system
echo.

python ble_enhanced_terminal.py

echo.
echo System stopped.
pause

