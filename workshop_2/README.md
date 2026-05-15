# Workshop 2 — Machine Learning & Deep Learning Aplicado

**Universidad EAFIT – Introducción a la Inteligencia Artificial**

Este proyecto integra dos problemas de aprendizaje supervisado:
1. **Clasificación**: Detección de fatiga muscular en ciclistas mediante señales EMG.
2. **Regresión**: Estimación de edad a partir de imágenes faciales (UTKFace).

---

## Estructura del Proyecto

```text
workshop_2/
├── README.md                  ← Este archivo (introducción)
├── Manual.md                  ← Guía técnica paso a paso
├── clasificacion/
│   └── clasificacion.ipynb    ← Notebook con RandomForest (F1: ~0.99)
└── regresion/
    └── regresion.ipynb        ← Notebook con PyTorch CNN (R2: ~0.55)
```

## Resumen de Resultados

### Problema 1: Clasificación (Fatiga Muscular)
- **Modelos:** Random Forest, kNN, Gradient Boosting, DNN.
- **Mejor Modelo:** Random Forest (Accuracy: ~90%, F1-Score Weighted: ~0.90).
- **Nota:** Se observó un overfitting marcado en el entrenamiento (100% de métricas), pero el modelo generalizó con un sólido 90% en el reentrenamiento final.
- **Conclusión:** Las características espectrales (frecuencia media y mediana) son determinantes para identificar el estado de fatiga.

### Problema 2: Regresión (Edad)
- **Modelo:** Deep CNN en PyTorch (4 bloques conv + BatchNorm + Dropout).
- **Dataset:** UTKFace (~24,106 imágenes).
- **Métricas Finales:**
  - **MAE (Test):** 9.51 años.
  - **R2 Score:** 0.5705.
- **Conclusión:** El modelo es altamente eficaz identificando etapas de vida (niñez vs adultez), logrando una generalización estable mediante un R2 del 57% y regularización activa.

---
**Desarrollado con PyTorch & Lightning.ai T4 GPU.**
