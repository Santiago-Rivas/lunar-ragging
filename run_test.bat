@echo off
REM Create tests directory if it doesn't exist
if not exist tests mkdir tests

REM Create output directory
if not exist output mkdir output
if not exist output\tests mkdir output\tests

REM Ensure test config exists
if not exist tests\config.json (
  echo Error: tests\config.json not found
  exit /b 1
)

REM Generate timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get LocalDateTime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

REM Run tests with the sample configuration
python src\rag_refactored.py test ^
  --config tests\config.json ^
  --output output\tests\test_results_%TIMESTAMP%.csv ^
  --dossier_dir data\space_data\dossiers ^
  --html_dir data\html ^
  --max_workers 6 