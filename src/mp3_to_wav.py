"""Конвертация MP3 в WAV: моно, PCM 16 бит, по умолчанию 44 100 Гц.

Без аргументов обрабатывает data/clotho/manual/audio/ и сохраняет WAV рядом
с исходными MP3. Существующие WAV пропускает, MP3 сохраняет.

Зависимости: python -m pip install numpy scipy soundfile

Примеры:
    python src/mp3_to_wav.py
    python src/mp3_to_wav.py --output-dir data/clotho/manual/audio_wav
    python src/mp3_to_wav.py --input-dir path/to/mp3 --sample-rate 48000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "clotho" / "manual" / "audio"


def convert(source: Path, destination: Path, sample_rate: int) -> None:
    import soundfile as sf
    from scipy.signal import resample_poly

    samples, original_rate = sf.read(source, dtype="float32", always_2d=True)
    if not len(samples):
        raise ValueError("пустая аудиозапись")
    mono = samples.mean(axis=1)
    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, original_rate // divisor)

    # Незавершённый файл не должен остаться под именем готового WAV.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".wav", delete=False) as tmp:
        temporary_path = Path(tmp.name)
    try:
        sf.write(temporary_path, mono, sample_rate, format="WAV", subtype="PCM_16")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="папка с MP3 (по умолчанию data/clotho/manual/audio)")
    parser.add_argument("--output-dir", type=Path,
                        help="папка для WAV (по умолчанию рядом с MP3)")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="частота дискретизации WAV, Гц (по умолчанию 44100)")
    args = parser.parse_args()
    if args.sample_rate <= 0:
        parser.error("--sample-rate должен быть положительным")
    if not args.input_dir.is_dir():
        parser.error(f"нет папки: {args.input_dir}")

    sources = sorted(path for path in args.input_dir.iterdir()
                     if path.is_file() and path.suffix.lower() == ".mp3")
    if not sources:
        print(f"MP3 не найдены: {args.input_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = skipped = errors = 0
    for source in sources:
        destination = output_dir / f"{source.stem}.wav"
        if destination.is_file():
            print(f"Пропуск: {destination.name} уже существует")
            skipped += 1
            continue
        try:
            convert(source, destination, args.sample_rate)
        except ImportError as exc:
            print(f"Не хватает зависимости: {exc}.\n"
                  "Установите: python -m pip install numpy scipy soundfile", file=sys.stderr)
            return 2
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Ошибка: {source.name}: {exc}", file=sys.stderr)
            errors += 1
            continue
        print(f"Готово: {source.name} -> {destination.name}")
        converted += 1

    print(f"\nСоздано: {converted}, пропущено: {skipped}, ошибок: {errors}. "
          f"WAV: {output_dir.resolve()}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
