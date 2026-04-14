@echo off
echo ========================================
echo   Testing Complete Integration Flow
echo ========================================
echo.

set BASE_URL=http://localhost:4000/api/v1
set EMAIL=test_%RANDOM%@example.com
set PASSWORD=Test123!
set FIRST_NAME=Test
set LAST_NAME=User

echo [1/5] Registering new user: %EMAIL%
echo.

curl -s -X POST "%BASE_URL%/auth/register" ^
  -H "Content-Type: application/json" ^
  -d "{\"email\":\"%EMAIL%\",\"password\":\"%PASSWORD%\",\"firstName\":\"%FIRST_NAME%\",\"lastName\":\"%LAST_NAME%\"}" > temp_response.json

type temp_response.json
echo.
echo.

REM Extract token using PowerShell
for /f "tokens=*" %%i in ('powershell -Command "(Get-Content temp_response.json | ConvertFrom-Json).accessToken"') do set TOKEN=%%i

if "%TOKEN%"=="" (
    echo ERROR: Failed to get authentication token
    pause
    exit /b 1
)

echo [2/5] Registration successful!
echo Token: %TOKEN:~0,50%...
echo.

echo [3/5] Telling AI: "My name is %FIRST_NAME%"
echo.

curl -s -X POST "%BASE_URL%/chat" ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer %TOKEN%" ^
  -d "{\"message\":\"My name is %FIRST_NAME%\"}" > temp_chat1.json

type temp_chat1.json
echo.
echo.

timeout /t 2 /nobreak >nul

echo [4/5] Asking AI: "What is my name?"
echo.

curl -s -X POST "%BASE_URL%/chat" ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer %TOKEN%" ^
  -d "{\"message\":\"What is my name?\"}" > temp_chat2.json

type temp_chat2.json
echo.
echo.

echo [5/5] Getting chat history
echo.

curl -s -X GET "%BASE_URL%/chat/history" ^
  -H "Authorization: Bearer %TOKEN%" > temp_history.json

type temp_history.json
echo.
echo.

echo ========================================
echo   Test Complete!
echo ========================================
echo.
echo If you see responses above, the integration is working!
echo The AI should have remembered your name.
echo.

REM Cleanup temp files
del temp_response.json temp_chat1.json temp_chat2.json temp_history.json

pause
