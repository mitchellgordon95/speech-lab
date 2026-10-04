import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(description="Speech Lab — a local pronunciation workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8767)
    sub.add_parser("seed", help="Create original synthetic signals, not pronunciation references")
    bench = sub.add_parser("benchmark", help="Download and run real model inference on a synthetic fixture")
    bench.add_argument(
        "--models",
        nargs="+",
        default=["wavlm-base", "wavlm-large", "xls-r", "qwen-asr", "qwen-asr-large", "qwen-omni"],
    )
    bench.add_argument("--articulation", action="store_true")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("speechlab.app:app", host="127.0.0.1", port=args.port)
    elif args.command == "seed":
        from demos.fixtures import seed

        print(json.dumps({"clips": len(seed())}))
    else:
        from demos.articulation import analyze
        from demos.fixtures import seed
        from speechlab.audio import load
        from speechlab.config import DATA
        from speechlab.models import MODELS, extract_frames, unload

        clip = seed()[0]
        waveform, _ = load(clip["id"], sr=16000)
        report = []
        for name in args.models:
            t = time.monotonic()
            try:
                frames, backend = extract_frames(waveform, name, progress=lambda s: print(s, flush=True))
                first_seconds = time.monotonic() - t
                warm_start = time.monotonic()
                frames, backend = extract_frames(waveform, name)
                report.append(
                    {
                        "model": name,
                        "ok": True,
                        "revision": MODELS[name]["revision"],
                        "frames": len(frames),
                        "dimensions": frames.shape[-1],
                        "device": backend,
                        "audio_seconds": len(waveform) / 16000,
                        "load_and_first_inference_seconds": round(first_seconds, 3),
                        "warm_inference_seconds": round(time.monotonic() - warm_start, 3),
                    }
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                report.append({"model": name, "ok": False, "error": str(e)})
            print(
                json.dumps({"model": name, "elapsed": time.monotonic() - t, "ok": report[-1]["ok"]}),
                flush=True,
            )
            unload()
        if args.articulation:
            try:
                t = time.monotonic()
                result = analyze(clip["id"], progress=print)
                report.append(
                    {
                        "model": "sparc",
                        "ok": True,
                        "frames": len(result["times"]),
                        "device": result["device"],
                        "load_and_inference_seconds": round(time.monotonic() - t, 3),
                    }
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                report.append({"model": "sparc", "ok": False, "error": str(e)})
        (DATA / "benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if any(not r["ok"] for r in report):
            raise SystemExit(1)
