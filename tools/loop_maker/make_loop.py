#!/usr/bin/env python3
"""
Encuentra automaticamente una seccion de una cancion (tipicamente el coro)
que se repite musicalmente y la exporta como un loop de audio sin corte
audible, sin exceder una duracion maxima (por defecto 30s).

Metodo:
  1. Decodificar el mp3/audio de entrada a WAV sin perdidas (ffmpeg) para
     tener acceso a muestras exactas al momento de recortar.
  2. Analizar el audio (mono, 22050 Hz) con librosa:
       - croma (chroma_cqt) para comparar contenido armonico/melodico.
       - beat tracking + onset strength para alinear los cortes al pulso.
       - RMS para preferir secciones con mas energia (coros suelen sonar
         mas fuerte/lleno que las estrofas).
  3. Buscar el par (t0, t0+lag) con mayor similitud de croma, con
     min_duration <= lag <= max_duration. Esto encuentra un fragmento que
     literalmente se repite a si mismo mas adelante en la cancion -> es el
     mejor candidato para loopear, porque la propia grabacion demuestra que
     ese contenido se repite de forma natural.
  4. Ajustar (snap) t0 y t1 al pulso/onset mas cercano para que el corte
     caiga justo en un golpe ritmico en vez de a mitad de una nota/palabra.
  5. Extraer el segmento [t0, t1) del WAV original (muestras exactas).
  6. Crossfade "de loop": se toma un poco de audio real que continua
     despues de t1 (la continuacion natural, sin cortes) y se mezcla con
     el inicio del segmento. Esto hace que el ultimo sample del loop fluya
     hacia el primer sample del loop de forma identica a como fluye en la
     grabacion original, eliminando el click/salto de amplitud del corte.
  7. Exportar:
       - <nombre>_loop.mp3   -> el loop final (<= max_duration segundos).
       - <nombre>_demo_x4.mp3 -> el mismo loop repetido N veces seguidas,
         para verificar de oido que el corte no se nota.

Uso:
    python3 make_loop.py cancion.mp3
    python3 make_loop.py cancion.mp3 --max-duration 30 --outdir salida/
    python3 make_loop.py cancion.mp3 --search-start 120 --search-end 260

Requisitos: ffmpeg en PATH, y `pip install librosa numpy soundfile`.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import librosa
import soundfile as sf


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def decode_to_wav(input_path, wav_path):
    run(["ffmpeg", "-y", "-i", input_path, "-acodec", "pcm_s24le", wav_path])


def analyze(y, sr, hop_length=512):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = librosa.util.normalize(chroma, axis=0)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, trim=False, onset_envelope=onset_env
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    return chroma, times, beat_times, rms, onset_env, onset_times


def find_best_repeat(chroma, times, rms, min_duration, max_duration,
                      search_start, search_end, energy_weight,
                      length_weight=0.35, coarse_step=0.25, cmp_window=3.0):
    """Busca (t0, lag) que maximice similitud de croma entre [t0,t0+w] y
    [t0+lag, t0+lag+w], con min_duration <= lag <= max_duration.

    length_weight favorece loops mas cercanos a max_duration (aprovechar el
    presupuesto de segundos disponible) en vez de quedarse siempre con el
    fragmento repetido mas corto solo por tener una similitud levemente
    mayor."""
    sr_frame = times[1] - times[0] if len(times) > 1 else 0.02
    w_frames = max(1, int(cmp_window / sr_frame))
    dur = times[-1]
    search_end = min(search_end, dur - max_duration - cmp_window)
    if search_end <= search_start:
        search_end = max(search_start, dur - max_duration - cmp_window)

    rms_norm = rms / (rms.max() + 1e-9)

    def frame_at(t):
        return int(round(t / sr_frame))

    best = None
    t0 = search_start
    while t0 <= search_end:
        f0 = frame_at(t0)
        if f0 + w_frames >= chroma.shape[1]:
            break
        a = chroma[:, f0:f0 + w_frames]
        na = np.linalg.norm(a)
        energy_bonus = 0.6 + energy_weight * rms_norm[min(f0, len(rms_norm) - 1)]
        lag = min_duration
        while lag <= max_duration:
            f1 = frame_at(t0 + lag)
            if f1 + w_frames >= chroma.shape[1]:
                break
            b = chroma[:, f1:f1 + w_frames]
            sim = float(np.sum(a * b) / (na * np.linalg.norm(b) + 1e-9))
            length_bonus = (1 - length_weight) + length_weight * (lag / max_duration)
            score = sim * energy_bonus * length_bonus
            if best is None or score > best[0]:
                best = (score, sim, t0, lag)
            lag += coarse_step
        t0 += coarse_step

    return best  # (score, sim, t0, lag)


def refine_local(chroma, times, t0_guess, lag_guess, min_duration, max_duration,
                  window=1.5, step=0.02, cmp_window=1.5):
    sr_frame = times[1] - times[0] if len(times) > 1 else 0.02
    w_frames = max(1, int(cmp_window / sr_frame))

    def frame_at(t):
        return int(round(t / sr_frame))

    best = None
    for t0 in np.arange(max(0, t0_guess - window), t0_guess + window, step):
        f0 = frame_at(t0)
        if f0 + w_frames >= chroma.shape[1]:
            continue
        a = chroma[:, f0:f0 + w_frames]
        na = np.linalg.norm(a)
        for lag in np.arange(max(min_duration, lag_guess - window),
                              min(max_duration, lag_guess + window), step):
            f1 = frame_at(t0 + lag)
            if f1 + w_frames >= chroma.shape[1]:
                continue
            b = chroma[:, f1:f1 + w_frames]
            sim = float(np.sum(a * b) / (na * np.linalg.norm(b) + 1e-9))
            if best is None or sim > best[0]:
                best = (sim, t0, lag)
    return best  # (sim, t0, lag)


def snap_to_onset(t, onset_env, onset_times, window=0.6):
    mask = (onset_times > t - window) & (onset_times < t + window)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return t
    peak_idx = idxs[np.argmax(onset_env[idxs])]
    return float(onset_times[peak_idx])


def make_seamless_loop(wav_path, start_t, end_t, xfade_dur, out_wav_path):
    info = sf.info(wav_path)
    sr = info.samplerate
    start = int(round(start_t * sr))
    end = int(round(end_t * sr))
    xfade = int(round(xfade_dur * sr))
    seg_len = end - start

    with sf.SoundFile(wav_path) as f:
        f.seek(start)
        raw = f.read(seg_len + xfade, dtype="float64", always_2d=True)

    head = raw[:xfade].copy()
    tail_continuation = raw[seg_len:seg_len + xfade].copy()
    middle = raw[xfade:seg_len].copy()

    t = np.linspace(0, 1, xfade, endpoint=False)[:, None]
    fade_out = np.cos(t * np.pi / 2)  # peso de la continuacion real (1 -> 0)
    fade_in = np.sin(t * np.pi / 2)   # peso del inicio original (0 -> 1)
    blended_start = tail_continuation * fade_out + head * fade_in

    loop = np.concatenate([blended_start, middle], axis=0)
    sf.write(out_wav_path, loop, sr, subtype="PCM_24")

    seam_delta = np.abs(loop[-1] - loop[0])
    naive_delta = np.abs(raw[seg_len - 1] - raw[0])
    return loop.shape[0] / sr, seam_delta, naive_delta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Archivo de audio de entrada (mp3, wav, etc.)")
    ap.add_argument("--outdir", default=None, help="Carpeta de salida (default: junto al input)")
    ap.add_argument("--max-duration", type=float, default=30.0,
                    help="Duracion maxima del loop en segundos (default 30)")
    ap.add_argument("--min-duration", type=float, default=12.0,
                    help="Duracion minima del loop en segundos (default 12)")
    ap.add_argument("--search-start", type=float, default=None,
                    help="Segundo desde donde buscar el coro (default: 10%% de la cancion)")
    ap.add_argument("--search-end", type=float, default=None,
                    help="Segundo hasta donde buscar el coro (default: 95%% de la cancion)")
    ap.add_argument("--energy-weight", type=float, default=0.4,
                    help="Cuanto se favorecen las secciones con mas energia/volumen (0-1)")
    ap.add_argument("--length-weight", type=float, default=0.35,
                    help="Cuanto se favorecen loops mas largos (cercanos a max-duration) "
                         "sobre repeticiones mas cortas con similitud parecida (0-1)")
    ap.add_argument("--xfade", type=float, default=0.10,
                    help="Duracion del crossfade en segundos en el punto de union (default 0.10)")
    ap.add_argument("--demo-repeats", type=int, default=4,
                    help="Veces que se repite el loop en el archivo demo (default 4)")
    ap.add_argument("--bitrate", default="256k", help="Bitrate del mp3 de salida")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.input))[0]
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input)) or "."
    os.makedirs(outdir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        full_wav = os.path.join(tmp, "full.wav")
        print(f"[1/5] Decodificando '{args.input}' a WAV...", file=sys.stderr)
        decode_to_wav(args.input, full_wav)

        print("[2/5] Analizando croma, beats y energia...", file=sys.stderr)
        y, sr = librosa.load(full_wav, sr=22050, mono=True)
        dur = len(y) / sr
        chroma, times, beat_times, rms, onset_env, onset_times = analyze(y, sr)

        search_start = args.search_start if args.search_start is not None else 0.10 * dur
        search_end = args.search_end if args.search_end is not None else 0.95 * dur

        print(f"[3/5] Buscando la mejor seccion repetida entre "
              f"{search_start:.1f}s y {search_end:.1f}s "
              f"(duracion candidata {args.min_duration:.0f}-{args.max_duration:.0f}s)...",
              file=sys.stderr)
        coarse = find_best_repeat(chroma, times, rms, args.min_duration, args.max_duration,
                                   search_start, search_end, args.energy_weight,
                                   args.length_weight)
        if coarse is None:
            print("No se encontro una seccion candidata; prueba ajustando "
                  "--search-start/--search-end o --min-duration.", file=sys.stderr)
            sys.exit(1)
        _, _, t0_guess, lag_guess = coarse

        sim, t0, lag = refine_local(chroma, times, t0_guess, lag_guess,
                                     args.min_duration, args.max_duration)
        t1 = t0 + lag

        t0_snapped = snap_to_onset(t0, onset_env, onset_times)
        t1_snapped = snap_to_onset(t1, onset_env, onset_times)
        if t1_snapped - t0_snapped > args.max_duration:
            t1_snapped = t0_snapped + args.max_duration
        if t1_snapped - t0_snapped < args.min_duration:
            t0_snapped, t1_snapped = t0, t1  # fallback sin snap si el ajuste rompe el rango

        print(f"    -> inicio={t0_snapped:.2f}s  fin={t1_snapped:.2f}s  "
              f"duracion={t1_snapped - t0_snapped:.2f}s  similitud={sim:.3f}",
              file=sys.stderr)

        print("[4/5] Extrayendo y aplicando crossfade de loop...", file=sys.stderr)
        loop_wav = os.path.join(tmp, "loop.wav")
        final_len, seam_delta, naive_delta = make_seamless_loop(
            full_wav, t0_snapped, t1_snapped, args.xfade, loop_wav
        )
        print(f"    -> duracion final: {final_len:.2f}s  "
              f"(salto en la union: {seam_delta.mean():.4f} vs corte sin crossfade: "
              f"{naive_delta.mean():.4f})", file=sys.stderr)

        print("[5/5] Exportando mp3...", file=sys.stderr)
        loop_mp3 = os.path.join(outdir, f"{base}_loop.mp3")
        demo_mp3 = os.path.join(outdir, f"{base}_demo_x{args.demo_repeats}.mp3")
        run(["ffmpeg", "-y", "-i", loop_wav, "-c:a", "libmp3lame", "-b:a", args.bitrate, loop_mp3])
        run(["ffmpeg", "-y", "-stream_loop", str(args.demo_repeats - 1), "-i", loop_wav,
             "-c:a", "libmp3lame", "-b:a", args.bitrate, demo_mp3])

    print(f"\nListo:\n  {loop_mp3}\n  {demo_mp3}")


if __name__ == "__main__":
    main()
