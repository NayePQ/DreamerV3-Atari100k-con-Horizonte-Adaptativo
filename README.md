# DreamerV3 + Horizonte de Imaginacion Adaptativo (Fork)

Este repositorio es un **fork de DreamerV3** con una extension para control
adaptativo del horizonte de imaginacion durante entrenamiento en Atari100k
(caso principal: `atari100k_alien`).

## Base del repositorio

- Repositorio base: https://github.com/danijar/dreamerv3
- Autor original: Danijar Hafner
- Licencia original: MIT ([LICENSE](LICENSE))

Si usas este codigo, cita el trabajo original:

```bibtex
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## Innovacion implementada

DreamerV3 usa un horizonte fijo de imaginacion (`imag_length=15`). En este fork
se agrega un mecanismo opcional (`agent.imag_adapt.enabled`) que **atenua**
la contribucion de pasos imaginados cuando la perdida dinamica (`train/loss/dyn`)
indica mayor incertidumbre.

La idea es reducir gradientes ruidosos del actor en fases donde el modelo de
mundo esta menos confiable, sin cortar por completo la imaginacion.

### Regla adaptativa (resumen)

Sea `L_dyn` la perdida dinamica media del lote:

- EMA de dinamica:
  `ema_t = (1 - alpha) * ema_{t-1} + alpha * L_dyn`
- Ratio relativo:
  `r_t = L_dyn / max(eps, ema_t)`
- Activacion (estado "malo"):
  se activa cuando `step >= warmup` y `L_dyn > abs_thresh` y `r_t > rel_thresh`.
- Severidad:
  `sev = clip((r_t - rel_thresh) / (rel_max - rel_thresh), 0, 1)`
- Decaimiento:
  `gamma = 1 - sev * (1 - min_decay)`
- Mascara temporal:
  `m_k = 1` para `k < min_horizon`,
  luego `m_k = max(tail_floor, gamma^k)`.

Cuando no hay activacion, `m_k = 1` para todo `k`.

### Donde esta en el codigo

- Configuracion de la mejora: [dreamerv3/configs.yaml](dreamerv3/configs.yaml)
  - bloque `agent.imag_adapt`
- Implementacion de la mascara: [dreamerv3/agent.py](dreamerv3/agent.py)
  - metodo `_imag_mask(...)`
- Integracion en perdida de imaginacion: [dreamerv3/agent.py](dreamerv3/agent.py)
  - llamada `imag_loss(..., imag_mask=imag_mask, ...)`

## Instalacion

Requisitos:

- Python 3.11+
- JAX con CUDA (si se usa GPU)

Instalacion rapida:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U -r requirements.txt
```

Para JAX GPU, seguir guia oficial:
https://github.com/google/jax#pip-installation-gpu-cuda

## Ejecucion (Atari100k Alien, 200M)

### Baseline (sin mejora)

```bash
python dreamerv3/main.py \
  --logdir /ruta/a/artifacts/atari100k_alien_baseline/seed0_{timestamp} \
  --configs atari100k size200m \
  --task atari100k_alien \
  --script train \
  --jax.platform cuda \
  --jax.train_devices [0] \
  --jax.policy_devices [0] \
  --run.steps 110000 \
  --run.envs 1 \
  --run.train_ratio 256 \
  --agent.imag_adapt.enabled False
```

### Adapt (con mejora)

```bash
python dreamerv3/main.py \
  --logdir /ruta/a/artifacts/atari100k_alien_adapt/seed0_{timestamp} \
  --configs atari100k size200m \
  --task atari100k_alien \
  --script train \
  --jax.platform cuda \
  --jax.train_devices [0] \
  --jax.policy_devices [0] \
  --run.steps 110000 \
  --run.envs 1 \
  --run.train_ratio 256 \
  --agent.imag_adapt.enabled True \
  --agent.imag_adapt.abs_thresh 1.0 \
  --agent.imag_adapt.rel_thresh 1.15
```

Nota importante Atari100k:

- `run.steps = 110000` son pasos del agente.
- Con `repeat=4` en `env.atari100k`, eso corresponde aprox. a `400K`
  environment steps.

## Salidas y metricas

Cada corrida guarda:

- `metrics.jsonl`: metricas de entrenamiento
- `scores.jsonl`: retornos por episodio
- `scope/`: videos y trazas (incluye `report-openloop-image.mp4`)
- `ckpt/`: checkpoints

## Scripts auxiliares en este workspace

- Export de open-loop de mejor seed adapt:
  `artifacts/atari100k_alien_adapt/export_best_adapt_openloop.py`
- Export comparativo adapt vs baseline (incluye paneles y videos):
  `artifacts/atari100k_alien_adapt/export_openloop_5plus45_adapt_vs_baseline.py`

## Transparencia sobre datos de prueba

En `artifacts/atari100k_alien_adapt_prueba5/` hay material de **prototipado de
reporte**. Revisar `README_PRUEBA.md` antes de usar esos resultados como
benchmark final.

## Creditos

- DreamerV3 original: Danijar Hafner y colaboradores.
- Este fork agrega un mecanismo de horizonte de imaginacion adaptativo para
  analisis experimental en Atari100k.
