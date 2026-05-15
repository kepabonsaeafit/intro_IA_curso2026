$ErrorActionPreference = 'Stop'

$candidates = @(
    Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
)

$python = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $python) {
    throw "No se encontró Python dentro de 'venv' ni '.venv'. Activa o crea el entorno virtual primero."
}

& $python -m uvicorn app.main:app --reload @args
