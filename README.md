# certificados-personas-por-fecha-ingreso-provincia-localidad

Proyecto refactorizado para ejecutar experimentos de ML desde scripts Python (sin depender de notebooks) y generar reportes HTML por corrida.

## Estructura

- `src/`: módulos de datos, features, modelos, entrenamiento, evaluación, plots, reporting y orquestación.
- `scripts/run_experiment.py`: entrypoint de ejecución.
- `configs/baseline.yaml`: configuración base.
- `results/<run_id>/`: artefactos por ejecución (`config.yaml`, `metrics.json`, `predictions.csv`, `plots/`, `report.html`).

## Ejecución

```bash
python scripts/run_experiment.py --config configs/baseline.yaml
```

## Dependencias

- `pandas`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
