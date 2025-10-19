@echo off
echo Starting BLE Enhanced Terminal Visual Mapping System with Network Settings...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.13 and try again
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install Flask==3.0.0 Flask-CORS==4.0.0 paho-mqtt==1.6.1

REM Start the application
echo.
echo Starting BLE Enhanced Terminal Visual Mapping System...
echo Open your browser to: http://localhost:5000
echo.
python ble_network_settings_added.py

pause

