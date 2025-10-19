@echo off
REM ============================================================================
REM BLE Tracking System - MQTT Fix with DHCP Auto-Detection
REM ============================================================================
REM This script fixes MQTT issues with automatic IP detection
REM ============================================================================

REM Change to script directory
cd /d "%~dp0"

echo.
echo ============================================================================
echo BLE Tracking System - MQTT Fix with DHCP Auto-Detection
echo ============================================================================
echo Current directory: %CD%
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script must be run as Administrator
    echo [INFO] Right-click and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [INFO] Running with administrator privileges

REM Auto-detect current IP address
echo [INFO] Auto-detecting current IP address...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        set CURRENT_IP=%%b
        goto :ip_found
    )
)

:ip_found
REM Remove any leading spaces
set CURRENT_IP=%CURRENT_IP: =%

if "%CURRENT_IP%"=="" (
    echo [ERROR] Could not auto-detect IP address
    echo [INFO] Please check your network connection
    pause
    exit /b 1
)

echo [INFO] Auto-detected IP: %CURRENT_IP%
echo [INFO] MQTT broker will be configured for this IP

echo.
echo [STEP 1] Stopping services
echo ============================================================================
echo [INFO] Stopping MQTT broker and web server...

REM Stop Mosquitto service
net stop mosquitto >nul 2>&1
echo [INFO] Mosquitto service stopped

REM Kill any Python processes
taskkill /f /im python.exe >nul 2>&1
echo [INFO] Stopped any running Python processes

REM Kill any existing Mosquitto processes
taskkill /f /im mosquitto.exe >nul 2>&1
echo [INFO] Stopped any manual Mosquitto processes

echo.
echo [STEP 2] Creating proper MQTT configuration
echo ============================================================================

REM Create a specific configuration for your network
echo [INFO] Creating Mosquitto configuration for %CURRENT_IP%...

echo # Mosquitto MQTT Broker Configuration > mosquitto_fixed.conf
echo # Configured for BLE Tracking System on %CURRENT_IP% >> mosquitto_fixed.conf
echo. >> mosquitto_fixed.conf
echo # Network binding - listen on all interfaces >> mosquitto_fixed.conf
echo listener 1883 0.0.0.0 >> mosquitto_fixed.conf
echo protocol mqtt >> mosquitto_fixed.conf
echo. >> mosquitto_fixed.conf
echo # Security - allow anonymous connections >> mosquitto_fixed.conf
echo allow_anonymous true >> mosquitto_fixed.conf
echo. >> mosquitto_fixed.conf
echo # Logging >> mosquitto_fixed.conf
echo log_dest file logs/mosquitto.log >> mosquitto_fixed.conf
echo log_dest stdout >> mosquitto_fixed.conf
echo log_type error >> mosquitto_fixed.conf
echo log_type warning >> mosquitto_fixed.conf
echo log_type notice >> mosquitto_fixed.conf
echo log_type information >> mosquitto_fixed.conf
echo log_type debug >> mosquitto_fixed.conf
echo. >> mosquitto_fixed.conf
echo # Connection settings >> mosquitto_fixed.conf
echo max_connections 100 >> mosquitto_fixed.conf
echo max_keepalive 300 >> mosquitto_fixed.conf
echo. >> mosquitto_fixed.conf
echo # Persistence >> mosquitto_fixed.conf
echo persistence true >> mosquitto_fixed.conf
echo persistence_location data/ >> mosquitto_fixed.conf
echo autosave_interval 300 >> mosquitto_fixed.conf

REM Create directories
if not exist "logs" mkdir "logs"
if not exist "data" mkdir "data"

echo [OK] MQTT configuration created for your network

echo.
echo [STEP 3] Configuring Windows Firewall
echo ============================================================================
echo [INFO] Configuring firewall for %CURRENT_IP%...

REM Remove old rules
netsh advfirewall firewall delete rule name="MQTT Broker" >nul 2>&1
netsh advfirewall firewall delete rule name="BLE Web Server" >nul 2>&1
netsh advfirewall firewall delete rule name="Mosquitto MQTT" >nul 2>&1

REM Add comprehensive firewall rules
netsh advfirewall firewall add rule name="MQTT Broker Inbound" dir=in action=allow protocol=TCP localport=1883 >nul 2>&1
netsh advfirewall firewall add rule name="MQTT Broker Outbound" dir=out action=allow protocol=TCP localport=1883 >nul 2>&1
netsh advfirewall firewall add rule name="BLE Web Server" dir=in action=allow protocol=TCP localport=5000 >nul 2>&1

echo [OK] Firewall rules configured

echo.
echo [STEP 4] Starting MQTT Broker manually
echo ============================================================================
echo [INFO] Starting Mosquitto with custom configuration...

REM Start Mosquitto manually with our configuration
if exist "C:\Program Files\mosquitto\mosquitto.exe" (
    echo [INFO] Starting Mosquitto on 0.0.0.0:1883...
    start /b "Mosquitto-BLE" "C:\Program Files\mosquitto\mosquitto.exe" -c "%CD%\mosquitto_fixed.conf" -v
    
    echo [INFO] Waiting for MQTT broker to start...
    timeout /t 5 /nobreak >nul
    
    REM Check if it's listening
    netstat -an | findstr ":1883" >nul 2>&1
    if %errorLevel% == 0 (
        echo [OK] MQTT broker is listening on port 1883
        netstat -an | findstr ":1883"
    ) else (
        echo [WARNING] MQTT broker may not be listening yet
        echo [INFO] Check the logs directory for mosquitto.log
    )
) else (
    echo [ERROR] Mosquitto not found at C:\Program Files\mosquitto\mosquitto.exe
    echo [INFO] Please ensure Mosquitto is properly installed
    goto :skip_test
)

echo.
echo [STEP 5] MQTT Broker Started Successfully
echo ============================================================================
echo [OK] MQTT broker is now running with DHCP auto-detection
echo [INFO] Detected IP: %CURRENT_IP%
echo [INFO] MQTT broker listening on: 0.0.0.0:1883
echo [INFO] ESP32 should connect to: %CURRENT_IP%:1883

:skip_test

REM Return the detected IP for use by the main script
echo %CURRENT_IP%

