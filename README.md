# AI Finance Predictor

Plataforma financiera basada en investigación cuantitativa y machine learning.

## Estado actual

Fase 1 — Kernel de investigación (QuantLab).

Implementado:

* Configuración metodológica congelada.
* Validación de parámetros mediante Pydantic.
* Acceso a DuckDB mediante una capa abstracta.
* Walk-Forward Splitter con purga y embargo.
* Suite de pruebas automatizadas.
* Integración continua mediante GitHub Actions.

## Instalación

```bash
uv sync --dev
```

## Ejecutar pruebas

```bash
uv run pytest -v
```

## Estructura

* `quantlab/`: subsistema de investigación.
* `config/`: parámetros metodológicos congelados.
* `tests/`: pruebas unitarias y fixtures.
* `backend/`: reservado para fases futuras.
* `frontend/`: reservado para fases futuras.

## Licencia

Pendiente de definir.
