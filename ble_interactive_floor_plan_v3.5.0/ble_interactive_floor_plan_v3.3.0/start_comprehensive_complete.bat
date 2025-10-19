@echo off
echo ========================================
echo  BLE Comprehensive Complete System
echo  Interactive Floor Plan System v1.0.0
echo ========================================
echo.

echo [1/3] Installing Python dependencies...
pip install Flask==3.0.0 Flask-CORS==4.0.0 paho-mqtt==1.6.1

echo.
echo [2/3] Starting BLE Comprehensive Complete System...
echo.
echo System Features:
echo - Interactive Floor Plan Editor
echo - Dynamic Map Editor with Drawing Tools
echo - Furniture Library with Drag-and-Drop
echo - SVG/PNG Import Support
echo - Scalable Coordinate System
echo - Enhanced Terminal Logging
echo - Python 3.13 Compatible
echo.

echo [3/3] Launching application...
echo.
echo Opening browser at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the system
echo ========================================

python ble_comprehensive_complete.py

pause

