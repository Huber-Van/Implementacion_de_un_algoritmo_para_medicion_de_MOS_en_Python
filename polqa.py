#!/usr/bin/env python3
"""
POLQA-like (educational) — v4 (calibrado)
-----------------------------------------
Versión educativa inspirada en ITU-T P.863 (POLQA). NO implementa la
recomendación oficial, pero sigue su filosofía:

Según ITU-T P.863 (POLQA), un modelo de calidad objetiva debe:
- Usar una señal de referencia “ideal” y una señal degradada.
- Hacer alineamiento temporal/acústico robusto.
- Trabajar en un dominio psicoacústico (bandas críticas / Bark / loudness).
- Derivar indicadores de distorsión y ruido.
- Mapear indicadores → MOS-LQO (Listening Quality Objective), comparable
  a MOS subjetivo ITU-T P.800/P.830.

Este código:
- Respeta esas ideas conceptuales (requeridas por P.863 a nivel de modelo).
- Implementa los detalles concretos (filtros, parámetros, mapeo a MOS)
  con diseños propios, NO tomados de la implementación oficial.
"""

from __future__ import annotations
import numpy as np, math, sys, argparse, os, tempfile, subprocess, shutil
from dataclasses import dataclass
from typing import Tuple

# I/O de audio: usamos soundfile si existe, si no scipy.io.wavfile
try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    from scipy.io import wavfile
    HAVE_SF = False

from scipy.signal import butter, sosfiltfilt, resample_poly, stft, get_window


# ---------- I/O / detección y carga de audio ----------

def ensure_wav(path: str) -> str:
    """
    Garantiza que 'path' apunte a un archivo WAV.

    Relación con ITU-T P.863:
    - P.863 asume que el modelo trabaja con señales de audio ya decodificadas
      y con formato conocido (típicamente PCM lineal).
    - La recomendación NO especifica cómo manejar múltiples formatos de archivo;
      eso es responsabilidad de la implementación.

    Lo que hacemos aquí (diseño propio):
    1) Si la extensión es .wav/.wave → se toma directamente.
    2) Si la extensión es otra pero el header es RIFF/WAVE → lo tratamos como WAV.
       (caso típico: archivos WAV con extensión mal puesta, p.ej. .way)
    3) Si no parece WAV, usamos ffmpeg para decodificar a WAV mono temporal.
    """
    ext = os.path.splitext(path)[1].lower()

    # Caso estándar: extensión WAV
    if ext in (".wav", ".wave"):
        return path

    # Intento de detección por encabezado RIFF/WAVE (WAV camuflado)
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        if len(header) >= 12 and header[0:4] == b"RIFF" and header[8:12] == b"WAVE":
            return path
    except OSError:
        # Si no se puede leer, fallará más adelante
        pass

    # Si no es WAV, intentamos convertir con ffmpeg (decisión de implementación)
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError(
            "ffmpeg no está instalado o no está en el PATH. "
            "Es necesario para convertir formatos no-WAV (mp3/opus/m4a, etc.) a WAV."
        )

    tmp_dir = tempfile.gettempdir()
    base = os.path.basename(path)
    name_no_ext = os.path.splitext(base)[0]
    out_path = os.path.join(tmp_dir, f"{name_no_ext}_conv.wav")

    cmd = [
        ffmpeg_exe,
        "-y",              # sobrescribe si ya existe
        "-i", path,        # entrada
        "-ac", "1",        # forzamos mono (1 canal)
        out_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al convertir {path} a WAV con ffmpeg: {e}") from e

    if not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg no produjo el archivo WAV esperado: {out_path}")

    return out_path


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """
    Carga audio desde 'path' en float32 mono, normalizado a ~-26 dBov.

    Relación con ITU:
    - ITU-T P.56 (medición de nivel de potencia de habla) y P.863
      trabajan con niveles de referencia alrededor de -26 dBov para voz.
    - La recomendación P.863 exige un control de nivel, pero NO fija exacto
      cómo lo implementas en código.

    Lo que hacemos:
    - Convertimos a float32.
    - Promediamos canales si es estéreo (requerido conceptualmente por P.863
      cuando se evalúa un solo canal perceptual).
    - Eliminamos DC.
    - Normalizamos RMS a ~ -26 dBov (aprox.), en línea con el espíritu ITU.
    """
    wav_path = ensure_wav(path)

    if HAVE_SF:
        x, fs = sf.read(wav_path, dtype="float32", always_2d=False)
    else:
        from scipy.io import wavfile
        fs, x = wavfile.read(wav_path)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
        elif x.dtype == np.uint8:
            x = (x.astype(np.float32) - 128) / 128.0
        else:
            x = x.astype(np.float32)

    # Promediado a mono (P.863 se aplica sobre uno o dos canales; aquí 1 canal)
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Quitamos componente DC y normalizamos nivel
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    if rms > 0:
        target = 10 ** (-26 / 20)  # ≈ -26 dBov
        x = x * (target / rms)

    return x.astype(np.float32), int(fs)


# ---------- Filtros y preprocesado en banda telefónica ----------

def bandpass_sos(fs: int, lo=300.0, hi=3400.0, order=4):
    """
    Filtro pasa banda simple (Butterworth) para aproximar banda telefónica.

    Relación con P.863:
    - POLQA define distintas bandas para narrowband, wideband, etc., con
      filtros específicos.
    - Aquí solo hacemos una aproximación: banda de voz clásica (~300–3400 Hz).
      Esto respeta el concepto ITU (analizar en banda de voz), pero no usa
      los filtros exactos de la recomendación.
    """
    ny = 0.5 * fs
    lo = max(10.0, lo) / ny
    hi = min(0.99 * ny, hi) / ny
    return butter(order, [lo, hi], btype="band", output="sos")


def prefilter_align(x: np.ndarray, fs: int, mode: str):
    """
    Prefiltrado previo al alineamiento:
    - En P.863 es obligatorio tener un preprocesamiento que limite la banda
      según el modo (NB, WB, SWB).
    - Aquí aplicamos un bandpass diferente para 'nb' y 'fb' como aproximación.

    mode:
      'nb' → aprox. banda [290, 3300] Hz (narrowband)
      'fb' → aprox. banda [320, 3400] Hz (fullband simplificado)
    """
    if mode == "nb":
        sos = bandpass_sos(fs, 290, 3300)
    else:
        sos = bandpass_sos(fs, 320, 3400)
    return sosfiltfilt(sos, x).astype(np.float32)


# ---------- VAD simple en tiempo (para alineamiento grueso) ----------

def frame_indices(n: int, win: int, hop: int):
    """Generador de índices [start, end) de frames en muestras."""
    for start in range(0, max(1, n - win + 1), hop):
        yield start, start + win


def simple_vad(x: np.ndarray, fs: int, win_ms=20, hop_ms=10, thr_db=-45):
    """
    VAD de energía sencillo, NO es el VAD oficial de P.863.

    Concepto ITU:
    - P.863 requiere una detección de actividad de voz (VAD) para ignorar
      partes no representativas (silencio, ruido puro).
    - El estándar no obliga a un algoritmo concreto; nuestra implementación
      es un VAD básico basado en energía.

    Lo que se hace:
    - Energía en dB por frame.
    - Se compara con mediana-6 dB y un umbral mínimo thr_db.
    """
    win = int(win_ms * fs / 1000)
    hop = int(hop_ms * fs / 1000)
    e = []
    for i0, i1 in frame_indices(len(x), win, hop):
        seg = x[i0:i1]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        e.append(10 * np.log10(np.mean(seg**2) + 1e-12))
    e = np.array(e)
    med = np.median(e)
    vad = (e > max(thr_db, med - 6)).astype(np.float32)
    return vad, e


# ---------- Feature auxiliar: dimensión fractal (solo para alineamiento) ----------

def fractal_dimension_sevcik(x: np.ndarray) -> float:
    """
    Dimensión fractal de Sevcik como descriptor de “rugosidad” temporal.

    Esto NO está en P.863. Es un truco local:
    - Añadir una segunda característica además de la energía para mejorar
      la robustez del alineamiento grueso.
    - Podría reemplazarse por otros features sin violar la filosofía ITU.
    """
    N = len(x)
    if N < 2:
        return 1.0
    y = (x - x.min()) / (x.max() - x.min() + 1e-12)
    L = np.sum(np.sqrt((1.0 / N) ** 2 + np.diff(y) ** 2))
    FD = 1 + (math.log(L) + math.log(2)) / math.log(2 * N)
    return float(FD)


# ---------- Alineamiento grueso + fino (concepto requerido por P.863) ----------

@dataclass
class AlignParams:
    """
    Parámetros del alineamiento grueso.

    Relación con ITU-T P.863:
    - La recomendación exige un alineamiento temporal muy robusto entre
      señal de referencia y señal degradada (sincronización).
    - El diseño exacto del algoritmo (Viterbi, correlación, etc.) es libre
      siempre que se logre un alineamiento adecuado.
    """
    win_ms: int = 32
    hop_ms: int = 16
    max_lag_ms: int = 200
    lag_step_ms: int = 4
    penalty_lambda: float = 1.0


def build_features(x: np.ndarray, fs: int, ap: AlignParams):
    """
    Extrae features por frame para alineamiento:
    - log-energía
    - dimensión fractal

    Esto implementa la parte de “medir similitud entre frames”
    requerida conceptualmente por P.863 (pero con features simples).
    """
    win = int(ap.win_ms * fs / 1000)
    hop = int(ap.hop_ms * fs / 1000)
    feats = []
    for i0, i1 in frame_indices(len(x), win, hop):
        seg = x[i0:i1]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        e = np.mean(seg**2)
        fd = fractal_dimension_sevcik(seg)
        feats.append([np.log(e + 1e-12), fd])
    F = np.asarray(feats, dtype=np.float32)
    F = F - F.mean(axis=0, keepdims=True)
    F = F / (F.std(axis=0, keepdims=True) + 1e-12)
    return F, win, hop


def zero_lag_bias_matrix(lags: np.ndarray, max_lag: int, alpha=0.15):
    """
    Sesgo que favorece retardos pequeños (lag≈0).

    Concepto ITU:
    - P.863 también impone restricciones de suavidad y continuidad
      en la trayectoria de alineamiento (no saltar retardos a lo loco).
    """
    pen = -alpha * (lags.astype(np.float32) / (max_lag + 1e-9)) ** 2
    return pen


def coarse_align(ref_filt: np.ndarray, deg_filt: np.ndarray, fs: int,
                 vad: np.ndarray, ap: AlignParams):
    """
    Alineamiento grueso estilo “path-finding”:

    - Calcula features para ref y deg.
    - Para frames marcados como voz, explora posibles lags y mide similitud.
    - Usa un esquema tipo Viterbi para encontrar el mejor camino de lags
      en el tiempo (trayectoria suave), como exige P.863 a nivel conceptual.

    NOTA:
    - El algoritmo concreto NO es el oficial de POLQA, pero cumple el rol
      de alinear las dos señales antes del análisis psicoacústico.
    """
    F_ref, win, hop = build_features(ref_filt, fs, ap)
    F_deg, _, _ = build_features(deg_filt, fs, ap)
    T = min(len(F_ref), len(F_deg), len(vad))
    F_ref, F_deg, vad = F_ref[:T], F_deg[:T], vad[:T]

    max_lag = int(ap.max_lag_ms * fs / 1000)
    lag_step = max(1, int(ap.lag_step_ms * fs / 1000))
    lags = np.arange(-max_lag, max_lag + 1, lag_step, dtype=int)
    L = len(lags)

    C = np.full((T, L), -np.inf, dtype=np.float32)
    for t in range(T):
        if vad[t] < 0.5:
            continue
        for li, lag in enumerate(lags):
            idx_ref = t
            idx_deg = t + int(round(lag / hop))
            if 0 <= idx_deg < T:
                num = float(np.dot(F_ref[idx_ref], F_deg[idx_deg]))
                den = float(
                    np.linalg.norm(F_ref[idx_ref]) * np.linalg.norm(F_deg[idx_deg]) + 1e-12
                )
                C[t, li] = num / den

    # Viterbi con penalización de cambios de lag y sesgo a lag pequeño
    bias = zero_lag_bias_matrix(lags, max_lag, alpha=0.15)
    DP = np.full_like(C, -np.inf)
    BP = np.full((T, L), -1, dtype=int)
    DP[0] = C[0] + bias
    for t in range(1, T):
        for li in range(L):
            prev = DP[t - 1] - ap.penalty_lambda * ((lags - lags[li]) ** 2) / (max_lag**2 + 1e-9)
            j = int(np.argmax(prev))
            DP[t, li] = C[t, li] + prev[j] + bias[li]
            BP[t, li] = j

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(DP[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = BP[t + 1, path[t + 1]] if BP[t + 1, path[t + 1]] >= 0 else path[t + 1]

    best_lags = lags[path]
    for t in range(T):
        if vad[t] < 0.5:
            best_lags[t] = best_lags[t - 1] if t > 0 else 0
    return best_lags, (win, hop)


def fine_align(ref: np.ndarray, deg: np.ndarray, fs: int,
               best_lags: np.ndarray, hop: int, search_ms=8):
    """
    Alineamiento fino:
    - Para cada frame, busca alrededor del lag grueso el retardo que maximiza
      la correlación entre energías (x^2) de ref y deg.

    Relación con P.863:
    - P.863 también realiza un refinamiento del alineamiento tras
      un alineamiento inicial.
    - La técnica concreta es decisión de implementación; aquí usamos
      correlación de energía.
    """
    T = len(best_lags)
    search = max(1, int(search_ms * fs / 1000))
    ref2 = ref.astype(np.float64)
    deg2 = deg.astype(np.float64)
    fine = np.zeros(T, dtype=int)
    for t in range(T):
        i0 = t * hop
        i1 = min(len(ref2), i0 + hop * 2)
        seg_r = ref2[max(0, i0 - hop):i1]
        lag0 = int(best_lags[t])
        lrange = range(lag0 - search, lag0 + search + 1)
        best_corr = -np.inf
        best = lag0
        for lag in lrange:
            j0 = max(0, i0 + lag - hop)
            j1 = min(len(deg2), j0 + len(seg_r))
            seg_d = deg2[j0:j1]
            sr = seg_r[:len(seg_d)]
            if len(sr) < 8:
                continue
            c = np.corrcoef(sr**2, seg_d**2)[0, 1]
            if np.isfinite(c) and c > best_corr:
                best_corr = c
                best = lag
        fine[t] = best
    return fine


def sample_rate_ratio(best_lags: np.ndarray, hop: int, fs: int, vad: np.ndarray) -> float:
    """
    Estima desajuste de frecuencia de muestreo (drift) entre ref y deg.

    Concepto ITU:
    - P.863 considera diferencias de sample rate y drift.
    - Aquí usamos una regresión lineal simple sobre los lags en regiones
      de voz para detectar si hay pendiente (drift).
    """
    idx = np.where(vad > 0.5)[0]
    if len(idx) < 5:
        return 1.0
    y = best_lags[idx].astype(np.float64)
    x = idx.astype(np.float64)
    xm, ym = x.mean(), y.mean()
    num = np.sum((x - xm) * (y - ym))
    den = np.sum((x - xm) ** 2) + 1e-12
    a = num / den
    b = ym - a * xm
    yhat = a * x + b
    resid = y - yhat
    ss_tot = np.sum((y - ym) ** 2) + 1e-12
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot
    span_ms = (y.max() - y.min()) / fs * 1000.0
    if r2 > 0.9 and span_ms >= 32.0:
        N = hop
        srratio = 1.0 - a / (N + 1e-12)
        return float(srratio)
    return 1.0


# ---------- Banco Bark y espectro psicoacústico ----------

def hz_to_bark(f_hz: np.ndarray) -> np.ndarray:
    """
    Conversión Hz → Bark.

    Relación con P.863:
    - POLQA trabaja en dominios psicoacústicos (bandas críticas, Bark/ERB).
    - La fórmula concreta de Bark es estándar, no impuesta por P.863, pero
      es totalmente coherente con el enfoque ITU.
    """
    f = np.maximum(f_hz, 1e-6)
    return 26.81 / (1 + 1960.0 / f) - 0.53


def bark_filterbank(fs: int, n_fft: int, n_bands: int = 24):
    """
    Banco de filtros triangulares en Bark.

    Concepto ITU:
    - Es obligatorio en P.863 tener un análisis en bandas críticas similar.
    - La cantidad de bandas y forma exacta de filtros aquí es elección propia,
      pero respeta la idea de trabajar en un “Bark-spectrogram”.
    """
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)
    bark = hz_to_bark(freqs)
    bmax = bark[-1]
    centers = np.linspace(1.0, min(24.0, bmax - 0.001), n_bands)
    H = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for i, c in enumerate(centers):
        left = centers[max(0, i - 1)] if i > 0 else c - 1.5
        right = centers[i + 1] if i < n_bands - 1 else c + 1.5
        H[i] = np.clip(
            np.minimum(
                (bark - left) / (c - left + 1e-9),
                (right - bark) / (right - c + 1e-9),
            ),
            0,
            1,
        )
    return H, freqs


def pitch_power_density(x: np.ndarray, fs: int, n_fft=1024, hop=256):
    """
    Densidad de potencia en Bark:

    - STFT (ventana Hann).
    - Potencia |Z|^2.
    - Proyección al banco Bark → Bark-spectrogram.
    - Log10 de la potencia.

    Esto implementa el “dominio psicoacústico espectro-temporal” que
    P.863 requiere conceptualmente, aunque la fórmula precisa sea distinta.
    """
    win = get_window("hann", n_fft, fftbins=True)
    f, t, Z = stft(
        x,
        fs=fs,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        padded=True,
        boundary=None,
    )
    P = (np.abs(Z) ** 2).astype(np.float32)
    H, _ = bark_filterbank(fs, n_fft)
    B = H @ P
    B = np.log10(B + 1e-12)
    return B


# ---------- VAD y recorte en rejilla STFT ----------

def stft_frames_energy(x: np.ndarray, fs: int, n_fft=1024, hop=256):
    """
    Energía por frame en la rejilla STFT.

    Relación con P.863:
    - La recomendación también realiza un gating de voz en el
      dominio tiempo-frecuencia, aunque el algoritmo exacto difiera.
    """
    win = get_window("hann", n_fft, fftbins=True)
    f, t, Z = stft(
        x,
        fs=fs,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        padded=True,
        boundary=None,
    )
    P = (np.abs(Z) ** 2).astype(np.float32)
    e = P.mean(axis=0)
    return e, P.shape[1]


def stft_vad_from_energy(e: np.ndarray, rel_db=-25.0):
    """
    VAD sobre energía en STFT:

    - Se pasa energía a dB.
    - Se usa un umbral relativo a la mediana (mediana + rel_db).

    Esto sigue el concepto ITU de “noise gating” alrededor de la voz,
    aunque la fórmula exacta sea nuestra.
    """
    e_db = 10 * np.log10(e + 1e-12)
    med = np.median(e_db)
    thr = med + rel_db
    vad = (e_db > thr).astype(np.float32)
    return vad


def find_start_stop_from_vad(vad: np.ndarray, min_on=3):
    """
    Encuentra inicio y fin del segmento principal de actividad de voz.

    Concepto ITU:
    - P.863 requiere evitar medir MOS sobre largas secciones de silencio.
    - Aquí detectamos el bloque continuo de voz más importante.
    """
    n = len(vad)
    start = 0
    cnt = 0
    for i in range(n):
        cnt = cnt + 1 if vad[i] > 0.5 else 0
        if cnt >= min_on:
            start = i - min_on + 1
            break
    cnt = 0
    stop = n - 1
    for i in range(n - 1, -1, -1):
        cnt = cnt + 1 if vad[i] > 0.5 else 0
        if cnt >= min_on:
            stop = i + min_on - 1
            break
    start = max(0, start)
    stop = min(n - 1, stop)
    if stop < start:
        start, stop = 0, n - 1
    return start, stop


# ---------- Reconstrucción alineada con Overlap-Add ----------

def overlap_add_align(xr: np.ndarray, yd: np.ndarray,
                      hop_samp: int, fine_lags: np.ndarray, ola_mul=2):
    """
    Reconstruye versión alineada de la degradada usando OLA (overlap-add).

    Relación con P.863:
    - POLQA aplica una especie de “time-warping” para alinear finamente
      la degradada con la referencia antes de calcular los mapas de
      disturbancia.
    - Aquí hacemos un time-warp simplificado usando segmentos desplazados
      y ventana Hann.
    """
    L = int(ola_mul * hop_samp)
    L = max(L, hop_samp)
    w = get_window("hann", L, fftbins=True).astype(np.float32)
    out = np.zeros_like(xr[: (len(fine_lags) * hop_samp + (L - hop_samp))])
    acc = np.zeros_like(out)
    T = len(fine_lags)
    for t in range(T):
        i0 = t * hop_samp
        j0 = i0 + int(fine_lags[t])
        seg_d = np.zeros(L, dtype=np.float32)
        if j0 < len(yd) and j0 + L > 0:
            s0 = max(0, j0)
            s1 = min(len(yd), j0 + L)
            d0 = s0 - j0
            d1 = d0 + (s1 - s0)
            seg_d[d0:d1] = yd[s0:s1]
        o0 = i0
        if o0 >= len(out):
            break
        o1 = min(i0 + L, len(out))
        seg_d = seg_d[: o1 - o0]
        w_use = w[: o1 - o0]
        out[o0:o1] += seg_d * w_use
        acc[o0:o1] += w_use
    acc = np.maximum(acc, 1e-8)
    out = out / acc
    out = out[: len(xr)]
    return out


# ---------- Compensación lenta de ganancia ----------

def gain_variation_compensation(B_ref: np.ndarray, B_deg: np.ndarray, k=0.2) -> np.ndarray:
    """
    Compensa diferencias lentas de nivel entre referencia y degradada.

    Concepto ITU:
    - Los modelos ITU suelen normalizar loudness para no penalizar
      simplemente cambios globales de volumen.
    - Aquí se estima la diferencia media de nivel espectral por frame
      y se corrige parcialmente (factor k).
    """
    g = B_ref.mean(axis=0) - B_deg.mean(axis=0)
    if g.size >= 5:
        kernel = np.ones(5) / 5.0
        g = np.convolve(g, kernel, mode="same")
    B_deg2 = B_deg + k * g
    return B_deg2


# ---------- Indicadores + mapeo MOS calibrado (nuestro diseño) ----------

def indicators_and_mos_poly(B_ref: np.ndarray, B_deg: np.ndarray,
                            vad_frames: np.ndarray, mode: str):
    """
    Calcula indicadores (disturb, freq, noise) y los mapea a MOS_like y MOS_LQO.

    Relación con ITU-T P.863:
    - P.863 define varios indicadores internos (disturbance, noise, etc.)
      y luego una cadena de funciones no lineales que llevan a MOS-LQO.
    - Esta función respeta ese esquema conceptual, pero TODOS los detalles
      (fórmulas, constantes, pesos) son diseño nuestro, no del estándar.

    Objetivo de calibración (según tu criterio subjetivo):
    - ref vs ref (sn15 vs sn15) → MOS_LQO ≈ 4.5 (excelente).
    - ref vs sn10/sn5        → MOS_LQO en rango 3–4 (aceptable/bueno).
    - ref vs sn0             → MOS_LQO ≈ 2–2.5 (malo/regular).
    - ref vs otra frase      → MOS_LQO ≈ 1.5–2 (malo).
    """

    # --- 1) Frames de voz según VAD (requerido conceptualmente por P.863) ---
    idx = np.where(vad_frames > 0.5)[0]
    if len(idx) < 3:
        idx = np.arange(B_ref.shape[1])

    # --- 2) FREQ: coloración espectral de largo plazo ---
    # Idea similar a comparar long-term spectral envelopes.
    R = B_ref[:, idx].mean(axis=1)
    D = B_deg[:, idx].mean(axis=1)
    Rn = R - np.mean(R)
    Dn = D - np.mean(D)
    freq_ind = float(np.mean(np.abs(Dn - Rn)))

    # --- 3) NOISE: ruido adicional en las pausas de la referencia ---
    # P.863 también define métricas de ruido en tramos de no voz.
    idx_n = np.where(vad_frames <= 0.5)[0]
    if len(idx_n) >= 3:
        noise_ind = float(max(0.0, B_deg[:, idx_n].mean() - B_ref[:, idx_n].mean()))
    else:
        noise_ind = 0.0

    # --- 4) DISTURB: diferencia espectro-temporal suavizada ---
    # Inspirado en los mapas de disturbancia de P.863, pero muy simplificado.
    E = np.abs(B_deg - B_ref)
    if E.shape[1] >= 3:
        kernel = np.ones(3) / 3.0
        E = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, E)
    disturb = float(np.mean(E[:, idx]))

    # --- 5) Similitud espectral en Bark (corr_bark) ---
    # No aparece como tal en P.863, pero refleja si el contenido (forma del espectro)
    # es similar (misma frase) o muy distinto (otra frase / música, etc.).
    X = B_ref[:, idx].ravel()
    Y = B_deg[:, idx].ravel()
    if np.all(np.isfinite(X)) and np.all(np.isfinite(Y)) and np.std(X) > 1e-9 and np.std(Y) > 1e-9:
        corr_bark = float(np.corrcoef(X, Y)[0, 1])
    else:
        corr_bark = 1.0
    corr_bark = max(-1.0, min(1.0, corr_bark))

    # --- 6) Normalización saturante (x → [0,1]) ---
    # P.863 también aplica funciones no lineales a indicadores (pero distintas).
    def sat(x, c):
        """
        Función saturante tipo x/(x+c):
        - x pequeño  → ~0
        - x muy grande → ~1
        c controla cuán rápido llega a 1 (calibración propia).
        """
        x = max(0.0, float(x))
        return x / (x + c)

    # Constantes de saturación ajustadas para:
    # - Ser más suavemente castigador con ruido moderado (disturb, freq).
    # - Mantener sensibilidad a cambios de contenido (corr).
    c_disturb = 0.6
    c_freq    = 0.35
    c_noise   = 0.20
    c_corr    = 0.25

    d_corr   = 1.0 - corr_bark
    x_disturb = sat(disturb, c_disturb)
    x_freq    = sat(freq_ind, c_freq)
    x_noise   = sat(noise_ind, c_noise)
    x_corr    = sat(d_corr, c_corr)

    # --- 7) Combinación ponderada en “pérdidas de MOS” ---
    # Esta parte corresponde al “modelo perceptual” calibrado.
    # En P.863 los pesos y funciones son resultado de un fitting con grandes
    # bases de datos subjetivos; aquí lo hemos calibrado a partir de tu criterio.
    if mode == "nb":
        mos_cap = 4.5
    else:
        mos_cap = 4.8

    w_disturb = 1.0
    w_freq    = 0.8
    w_noise   = 0.7
    w_corr    = 1.8  # peso alto para castigar contenido distinto

    total_loss = (
        w_disturb * x_disturb +
        w_freq    * x_freq +
        w_noise   * x_noise +
        w_corr    * x_corr
    )

    mos_like = mos_cap - total_loss
    mos_like = float(np.clip(mos_like, 1.0, mos_cap))

    # --- 8) Re-escalado final a MOS_LQO ---
    # P.863 define MOS-LQO como MOS objetivo calibrado contra MOS subjetivo.
    # Aquí hacemos un re-escalado simple de [1, mos_cap] → [1, 4.5].
    lqo_cap = 4.5
    mos_lqo = 1.0 + (mos_like - 1.0) * (lqo_cap - 1.0) / (mos_cap - 1.0 + 1e-9)
    mos_lqo = float(np.clip(mos_lqo, 1.0, lqo_cap))

    return mos_lqo, mos_like, disturb, freq_ind, noise_ind


# ---------- Orquestación principal tipo POLQA-like ----------

def polqa_like(ref_path: str, deg_path: str, mode: str = "fb"):
    """
    Función principal (equivalente conceptual al pipeline de P.863):

    Requerido por ITU-T P.863 a nivel de modelo:
    - Uso de par referencia/degradado.
    - Alineamiento robusto.
    - Transformación psicoacústica.
    - Cálculo de indicadores de degradación.
    - Conversión a un MOS objetivo (MOS-LQO).

    Aquí se hace:
    1) Cargar y normalizar ref y deg.
    2) Prefiltrar según 'mode' (NB/FB).
    3) VAD sobre ref para alineamiento.
    4) Alineamiento grueso + fino (+ corrección de drift).
    5) Reconstrucción alineada de la degradada (time-warp).
    6) STFT en Bark + gating de voz.
    7) Compensación lenta de ganancia.
    8) Cálculo de indicadores y mapeo a MOS_like / MOS_LQO.
    """
    # --- Carga y re-muestreo si hace falta (ambas en la misma fs) ---
    xr, fsr = load_audio(ref_path)
    yd, fsd = load_audio(deg_path)
    if fsr != fsd:
        g = math.gcd(fsd, fsr)
        yd = resample_poly(yd, fsr // g, fsd // g).astype(np.float32)
        fsd = fsr

    # --- Prefiltros para alineamiento ---
    xr_f = prefilter_align(xr, fsr, "nb" if mode == "nb" else "fb")
    yd_f = prefilter_align(yd, fsr, "nb" if mode == "nb" else "fb")

    # --- VAD simple sobre referencia ---
    vad, _ = simple_vad(xr_f, fsr)

    # --- Alineamiento grueso + fino ---
    ap = AlignParams()
    best_lags, (win, hop) = coarse_align(xr_f, yd_f, fsr, vad, ap)
    fine = fine_align(xr_f, yd_f, fsr, best_lags, hop)

    # Chequeo de identidad temprana:
    # Si ref≈deg con correlación > 0.999 a lag 0, forzamos fine[:] = 0.
    Ncheck = min(len(xr_f), len(yd_f), int(2.0 * fsr))
    if Ncheck > fsr // 2:
        xr_chk = xr_f[:Ncheck]
        yd_chk = yd_f[:Ncheck]
        num = float(np.dot(xr_chk, yd_chk))
        den = float(np.linalg.norm(xr_chk) * np.linalg.norm(yd_chk) + 1e-12)
        if den > 0 and (num / den) > 0.999:
            fine[:] = 0

    # --- Corrección de drift de muestreo si se detecta ---
    srr = sample_rate_ratio(fine, hop, fsr, vad[: len(fine)])
    if abs(srr - 1.0) > 0.005:
        up = int(round(10000))
        down = int(round(10000 * srr))
        yd = resample_poly(yd, up, down).astype(np.float32)
        yd_f = prefilter_align(yd, fsr, "nb" if mode == "nb" else "fb")
        best_lags, (win, hop) = coarse_align(xr_f, yd_f, fsr, vad, ap)
        fine = fine_align(xr_f, yd_f, fsr, best_lags, hop)

    # --- Reconstrucción alineada de la degradada ---
    hop_samp = hop
    T = min(len(fine), max(8, (len(xr) // hop_samp) - 2, (len(yd) // hop_samp) - 2))
    if T < 8:
        T = len(fine)
    aligned_deg = overlap_add_align(xr, yd, hop_samp, fine[:T], ola_mul=2)

    # --- VAD en rejilla STFT + recorte de región activa ---
    e_frames, _ = stft_frames_energy(xr[: len(aligned_deg)], fsr)
    vad_spec = stft_vad_from_energy(e_frames, rel_db=-25.0)
    s0, s1 = find_start_stop_from_vad(vad_spec)

    # --- Espectros Bark en el tramo activo ---
    B_ref_full = pitch_power_density(xr[: len(aligned_deg)], fsr)
    B_deg_full = pitch_power_density(aligned_deg, fsr)
    B_ref = B_ref_full[:, s0 : s1 + 1]
    B_deg = B_deg_full[:, s0 : s1 + 1]
    vad_frames = vad_spec[s0 : s1 + 1]

    # --- Compensación de ganancia + mapeo a MOS ---
    B_deg2 = gain_variation_compensation(B_ref, B_deg)
    mos_lqo, mos_like, disturb, freq_ind, noise_ind = indicators_and_mos_poly(
        B_ref, B_deg2, vad_frames, mode
    )

    # Salida con campos pensados para JSON hacia tu app.
    return {
        "mos_lqo": mos_lqo,    # Este es el MOS que debes mostrar al usuario
        "mos_like": mos_like,  # Score interno para depuración/calibración
        "disturb": disturb,
        "freq": freq_ind,
        "noise": noise_ind,
    }


# ---------- CLI para pruebas por consola ----------

def main():
    """
    Uso por consola (para testear como en tus ejemplos):

        python polqa.py ref.wav deg.wav --mode fb

    Donde:
        ref = audio de referencia (la mejor versión, p.ej. sn15)
        deg = audio degradado (sn10, sn5, sn0, otra frase, etc.)
    """
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("deg")
    p.add_argument("--mode", choices=["nb", "fb"], default="fb")
    args = p.parse_args()
    out = polqa_like(args.ref, args.deg, mode=args.mode)
    print(f"Mode: {args.mode}")
    print(f"DISTURB={out['disturb']:.3f}  FREQ={out['freq']:.3f}  NOISE={out['noise']:.3f}")
    print(f"MOS_like: {out['mos_like']:.2f}  MOS_LQO: {out['mos_lqo']:.2f}")


if __name__ == "__main__":
    main()
