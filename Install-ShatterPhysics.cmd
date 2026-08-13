@echo off
rem Double-clickable: builds the plugin, then installs it (elevating if needed).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Build
