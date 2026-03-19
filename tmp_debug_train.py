"""Debug: Run train_real full pipeline and dump errors to a log file."""
import sys, os, traceback, io
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

log_path = "models/train_debug.log"
os.makedirs("models", exist_ok=True)

with open(log_path, "w") as logf:
    # Redirect stderr to file
    old_stderr = sys.stderr
    sys.stderr = logf

    try:
        from src.ml.train_real import train_robust_model
        train_robust_model()
        logf.write("\nSUCCESS\n")
    except Exception as e:
        logf.write(f"\nFAILED: {e}\n")
        traceback.print_exc(file=logf)
    finally:
        sys.stderr = old_stderr

print(f"Done. See {log_path}")
