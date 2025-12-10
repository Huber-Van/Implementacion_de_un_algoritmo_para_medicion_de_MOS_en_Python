# server.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil, tempfile, os, time
import soundfile as sf

# Importa tu implementación (debe estar en el mismo folder)
from polqa import polqa_like  # def polqa_like(ref_path, deg_path, mode='fb'|'nb')

app = FastAPI(title="POLQA-like API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Para demo en LAN; restringe en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PolqaResponse(BaseModel):
    mos_lqo: float
    indicators: dict
    align: dict
    eval: dict
    version: str
    timestamp: str

@app.get("/v1/ping")
def ping():
    return {"ok": True}

@app.post("/v1/polqa", response_model=PolqaResponse)
async def polqa_endpoint(
    ref: UploadFile = File(...),
    deg: UploadFile = File(...),
    mode: str = Form("fb")
):
    t0 = time.time()

    mode = mode.lower()
    if mode not in ("nb", "fb"):
        raise HTTPException(status_code=400, detail="mode debe ser 'nb' o 'fb'")

    tmpdir = tempfile.mkdtemp(prefix="polqa_")
    ref_path = os.path.join(tmpdir, f"ref_{ref.filename or 'ref'}")
    deg_path = os.path.join(tmpdir, f"deg_{deg.filename or 'deg'}")

    try:
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(ref.file, f)
        with open(deg_path, "wb") as f:
            shutil.copyfileobj(deg.file, f)

        def duration_sec(path: str) -> float:
            try:
                info = sf.info(path)
                if info.frames and info.samplerate:
                    return float(info.frames) / float(info.samplerate)
            except Exception:
                pass
            return 0.0

        ref_dur = duration_sec(ref_path)
        deg_dur = duration_sec(deg_path)

        out = polqa_like(ref_path, deg_path, mode=mode)
        mos = float(out.get("mos_lqo", out.get("pseudo_mos", 0.0)))
        disturb = float(out.get("disturb", 0.0))
        freq = float(out.get("freq", 0.0))
        noise = float(out.get("noise", 0.0))

        elapsed = time.time() - t0
        return PolqaResponse(
            mos_lqo = mos,
            indicators = {"disturb": disturb, "freq": freq, "noise": noise},
            align = {"srr": None, "lag_ms_mean": None, "lag_ms_std": None},
            eval = {
                "mode": mode,
                "duration_ref_s": round(ref_dur, 3),
                "duration_deg_s": round(deg_dur, 3),
                "proc_time_s": round(elapsed, 3)
            },
            version = "polqa-like-edu",
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
    finally:
        try:
            ref.file.close(); deg.file.close()
        except Exception:
            pass
        try:
            for p in (ref_path, deg_path):
                if os.path.exists(p): os.remove(p)
            os.rmdir(tmpdir)
        except Exception:
            pass
