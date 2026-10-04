import io
import hashlib
from functools import lru_cache

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

app = FastAPI(title="Nabra Arabic TTS")

MODEL_ID = "oddadmix/Nabra-82M-v0.1"


class SpeechRequest(BaseModel):
    text: str
    speed: float = 1.0


@lru_cache(maxsize=1)
def load_pipeline():
    from kokoro import KPipeline

    return KPipeline(
        lang_code="a",
        repo_id=MODEL_ID
    )


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "nabra-tts-service",
        "model": MODEL_ID
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/speech")
def speech(req: SpeechRequest):
    text = " ".join(req.text.split()).strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Missing text"
        )

    text = text[:1400]

    speed = min(
        1.3,
        max(0.7, float(req.speed))
    )

    try:
        pipeline = load_pipeline()

        chunks = []

        for _, _, audio in pipeline(
            text,
            voice="af_heart",
            speed=speed
        ):
            chunks.append(
                np.asarray(audio, dtype=np.float32)
            )

        if not chunks:
            raise RuntimeError("No audio generated")

        audio = np.concatenate(chunks)

        output = io.BytesIO()

        sf.write(
            output,
            audio,
            24000,
            format="WAV"
        )

        data = output.getvalue()

        etag = hashlib.sha256(
            (text + "|" + str(speed)).encode("utf-8")
        ).hexdigest()

        return Response(
            content=data,
            media_type="audio/wav",
            headers={
                "Cache-Control":
                    "public, max-age=31536000, immutable",
                "ETag": etag
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "TTS generation failed: "
                + type(error).__name__
            )
        )
