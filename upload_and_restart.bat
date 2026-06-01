@echo off
title ROBORDER-X Remote AWS Deployer & Restarter
chcp 65001 > nul

cls
echo ====================================================================
echo           ROBORDER-X Remote AWS Deployer & Restarter
echo ====================================================================
echo.

:: Detect bash path in a 100% crash-proof way (no nested parenthesis bugs)
set BASH_BIN=
if exist "C:\Program Files\Git\bin\bash.exe" (
    set "BASH_BIN=C:\Program Files\Git\bin\bash.exe"
)
if "%BASH_BIN%"=="" (
    if exist "C:\Program Files\Git\usr\bin\bash.exe" (
        set "BASH_BIN=C:\Program Files\Git\usr\bin\bash.exe"
    )
)
if "%BASH_BIN%"=="" (
    if exist "C:\Program Files\Git\git-bash.exe" (
        set "BASH_BIN=C:\Program Files\Git\git-bash.exe"
    )
)
if "%BASH_BIN%"=="" (
    where bash >nul 2>nul
    if not errorlevel 1 (
        set "BASH_BIN=bash"
    )
)

if "%BASH_BIN%"=="" (
    echo ====================================================================
    echo ERROR: Git Bash was not found on your system!
    echo Please make sure Git is installed at C:\Program Files\Git\
    echo ====================================================================
    echo.
    pause
    exit /b 1
)

echo Found Bash at: "%BASH_BIN%"
echo.
echo Starting deployment...
echo --------------------------------------------------------------------

:: Execute the scp and ssh commands sequentially inside Git's console bash
"%BASH_BIN%" -c "cd '%CD:\=/%' && echo 'Uploading config.py...' && scp -i D:/AI-Software/VPS-New/Eagle.pem src/config.py ubuntu@63.178.122.231:~/opt/ROBORDER/src/config.py && echo 'Uploading main.py...' && scp -i D:/AI-Software/VPS-New/Eagle.pem src/main.py ubuntu@63.178.122.231:~/opt/ROBORDER/src/main.py && echo 'Uploading trainer.py...' && scp -i D:/AI-Software/VPS-New/Eagle.pem src/agent/trainer.py ubuntu@63.178.122.231:~/opt/ROBORDER/src/agent/trainer.py && echo 'Uploading macro_news_schedule.json...' && scp -i D:/AI-Software/VPS-New/Eagle.pem macro_news_schedule.json ubuntu@63.178.122.231:~/opt/ROBORDER/macro_news_schedule.json && echo 'Uploading hybrid_engine.py...' && scp -i D:/AI-Software/VPS-New/Eagle.pem src/core/hybrid_engine.py ubuntu@63.178.122.231:~/opt/ROBORDER/src/core/hybrid_engine.py && echo 'Uploading pure_ppo_strategy.py...' && scp -i D:/AI-Software/VPS-New/Eagle.pem src/strategies/pure_ppo/pure_ppo_strategy.py ubuntu@63.178.122.231:~/opt/ROBORDER/src/strategies/pure_ppo/pure_ppo_strategy.py && echo 'Uploading dashboard_server.py...' && scp -i D:/AI-Software/VPS-New/Eagle.pem src/core/dashboard_server.py ubuntu@63.178.122.231:~/opt/ROBORDER/src/core/dashboard_server.py && echo 'Uploading index.html...' && scp -i D:/AI-Software/VPS-New/Eagle.pem static/index.html ubuntu@63.178.122.231:~/opt/ROBORDER/static/index.html && echo 'Uploading start_bot.sh...' && scp -i D:/AI-Software/VPS-New/Eagle.pem start_bot.sh ubuntu@63.178.122.231:~/opt/ROBORDER/start_bot.sh && echo 'Stopping old bot process...' && ssh -i D:/AI-Software/VPS-New/Eagle.pem ubuntu@63.178.122.231 'pkill -9 -f src.main 2>/dev/null || killall -9 python3 2>/dev/null || fuser -k 3000/tcp 2>/dev/null; sleep 3; echo Bot stopped successfully.' && echo 'Starting bot...' && ssh -i D:/AI-Software/VPS-New/Eagle.pem ubuntu@63.178.122.231 'nohup bash ~/opt/ROBORDER/start_bot.sh >/dev/null 2>&1 </dev/null & disown'"

echo.
echo ====================================================================
echo Deployment process finished. Press any key to close this window...
echo ====================================================================
pause > nul
