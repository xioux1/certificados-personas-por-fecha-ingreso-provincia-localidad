# certificados-personas-por-fecha-ingreso-provincia-localidad

## Aplicación del notebook de clase al caso del repositorio

Se implementó un pipeline completo en `mlp_tuning_pipeline.py` siguiendo la referencia de clase.

### Trazabilidad solicitada

- Carga y limpieza del dataset + conversión de tipos: **(extraido del notebook de clase)**.
- Construcción de `Pipeline` con escalado y MLP: **(extraido del notebook de clase)**.
- Validación cruzada con `cross_val_score`: **(extraido del notebook de clase)**.
- Generación de combinaciones con `ParameterGrid`: **(extraido del notebook de clase)**.
- Búsqueda exhaustiva con `GridSearchCV`: **(extraido del notebook de clase)**.
- Búsqueda aleatoria con `RandomizedSearchCV`: **(extraido del notebook de clase)**.
- Comparación de rendimiento base vs tuning: **(extraido del notebook de clase)**.

### Estado frente al pipeline previo del repositorio

No había un pipeline de ML previo en este repositorio (solo dataset + README), por lo tanto no aplica
la marca **(ya estaba implementado, coincide con notebook)** para componentes de entrenamiento/tuning.

### Resolución de conflictos

No se detectaron conflictos de implementación preexistente; se aplicó directamente la referencia de clase.

## Ejecución

```bash
python mlp_tuning_pipeline.py
```

> Requiere dependencias: `numpy`, `pandas`, `scikit-learn`.
