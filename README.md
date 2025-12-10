# POLQA educativo en Python (modelo MOS_LQO)

Implementación **educativa** inspirada en **ITU-T P.863 (POLQA)** para estimar la calidad de voz
en términos de **MOS_LQO (Listening Quality Objective)**, a partir de un par de señales:

- **Referencia (`ref`)**: señal “ideal”.
- **Degradada (`deg`)**: señal bajo prueba (ruido, códec, canal, etc.).

> ⚠️ Aviso  
> Este proyecto está inspirado en la arquitectura de ITU-T P.863, pero **no es** POLQA oficial
> ni pretende reemplazar herramientas certificadas. Se publica para fines académicos
> y experimentales.

---

## Características principales

- Implementado en **Python 3**.
- Acepta varios formatos de audio (WAV, OGG, MP3, etc.) usando `ffmpeg`.
- Normaliza el nivel de voz a ≈ **−26 dBov**.
- Alineamiento temporal **grueso + fino** entre referencia y degradada:
  - VAD (Voice Activity Detection) por energía.
  - Estimación de drift de frecuencia de muestreo.
  - Reconstrucción alineada mediante **OLA (Overlap-Add)**.
- Análisis psicoacústico:
  - STFT + banco de filtros en escala **Bark**.
  - Gating de voz en el dominio tiempo-frecuencia.
- Indicadores perceptuales:
  - `disturb`: diferencia espectro-temporal.
  - `freq`: coloración espectral.
  - `noise`: ruido en pausas.
  - `corr_bark`: similitud espectral global.
- Mapeo perceptual propio:
  - Indicadores normalizados → `MOS_like` → **`MOS_LQO`** (1 a ~4.5).

## Diagrama de bloques de la implementacion del algoritmo
ENTRADAS
────────
ref.wav → referencia (“ideal”)
deg.wav → señal degradada

PREPROCESADO
────────────
ref: Carga + Nivel ─┐
                    ├─ Prefiltrado NB/FB → VAD temporal (ref)
deg: Carga + Nivel ─┘

ALINEAMIENTO
────────────
Señales preprocesadas
      ↓
[5] Alineamiento grueso (similitud por frames)
      ↓
[6] Alineamiento fino + drift de fs
      ↓
[7] Time-Warp OLA (señal degradada alineada)

ANÁLISIS PSICOACÚSTICO
───────────────────────
[8] STFT + Bark → B_ref, B_deg
[9] VAD espectral → tramo activo
[10] Compensación de ganancia lenta → B_deg2

INDICADORES + MOS
─────────────────
[11] DISTURB, FREQ, NOISE, corr_bark
      ↓
[12] Mapeo perceptual → MOS_like → MOS_LQO

SALIDA
──────
{ MOS_LQO, MOS_like, DISTURB, FREQ, NOISE }

