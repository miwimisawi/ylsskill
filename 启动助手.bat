@echo off
chcp 65001 >nul
title 眼科住院助手

echo.
echo  ╔══════════════════════════════════════╗
echo  ║        眼科住院助手  启动中...       ║
echo  ╚══════════════════════════════════════╝
echo.

:: 激活 conda 环境
call conda activate openai 2>nul
if errorlevel 1 (
    echo [错误] 无法激活 conda 环境 "openai"
    echo 请确认已运行: conda create -n openai python=3.11
    pause
    exit /b 1
)

:: 切换到脚本所在目录（双击时工作目录可能不对）
cd /d "%~dp0"

:: 等待一秒后自动打开浏览器
start "" timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000/

echo  访问地址：
echo    用户界面   http://127.0.0.1:8000/
echo    调试界面   http://127.0.0.1:8000/debug.html
echo    API 文档   http://127.0.0.1:8000/docs
echo.
echo  日志文件：logs\app.log
echo  关闭本窗口即停止服务。
echo  ─────────────────────────────────────────
echo.

python run_server.py

echo.
echo  服务已停止。
pause
