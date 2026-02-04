import subprocess
import shutil
import os

# ================================
# TTS



def speak(text: str):
    text = text.strip()
    if not text:
        return

    try:
        if shutil.which("spd-say"):
            # -l ko 가 환경에 따라 안 먹을 수도 있어도, 대부분 안전하게 실행됨
            subprocess.run(["spd-say", "-l", "ko", text], check=False)
            return

        if shutil.which("pico2wave") and shutil.which("aplay"):
            wav_path = "/tmp/salute_tts.wav"
            subprocess.run(["pico2wave", "-l", "ko-KR", "-w", wav_path, text], check=False)
            subprocess.run(["aplay", "-q", wav_path], check=False)
            try:
                os.remove(wav_path)
            except OSError:
                pass
            return

        if shutil.which("espeak-ng"):
            subprocess.run(["espeak-ng", "-v", "ko", text], check=False)
            return

        if shutil.which("espeak"):
            subprocess.run(["espeak", "-v", "ko", text], check=False)
            return

    except Exception as e:
        print(f"TTS failed: {e}")
# ================================
