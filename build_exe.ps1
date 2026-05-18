Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'Virtual environment not found. Create .venv first and install the project dependencies.'
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --name Img-Tagbooru `
    --onefile `
    --windowed `
    --collect-all PySide6 `
    --add-data "frontend;frontend" `
    --add-data "backend;backend" `
    --add-data "danbooru_tags_post_count.csv;." `
    frontend\native\main_window.py
