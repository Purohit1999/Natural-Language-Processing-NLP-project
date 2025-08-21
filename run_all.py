import subprocess, sys, time
from pathlib import Path

# ✅ Only console scripts here (NO streamlit, NO run_all.py)
SCRIPTS = [
    "steaming_1.py",
    "lemmatizer_2.py",
    "spacy_lammatizer_3.py",
    "Project_steaming_and_lammatization_4.py",
    "bow_1.py",
    "bow_project_2.py",
    "tf-idf_1.py",
]

ROOT = Path(__file__).parent
TIMEOUT_SEC = 30  # per-script safety timeout

def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def run_script(name):
    path = ROOT / name
    if name == Path(__file__).name:
        return (name, "SKIP (self)")
    if not path.exists():
        return (name, "SKIP (missing)")
    if "streamlit" in name.lower():
        return (name, "SKIP (streamlit)")
    print("\n========== Running", name, "==========\n")
    t0 = time.time()
    try:
        subprocess.run([sys.executable, str(path)], check=True, timeout=TIMEOUT_SEC)
        return (name, f"OK ({time.time()-t0:.1f}s)")
    except subprocess.TimeoutExpired:
        return (name, f"TIMEOUT (> {TIMEOUT_SEC}s)")
    except subprocess.CalledProcessError as e:
        return (name, f"ERROR (exit {e.returncode})")

def main():
    results = [run_script(s) for s in unique(SCRIPTS)]
    print("\n----- Summary -----")
    for n, status in results:
        print(f"{n}: {status}")
    print("-------------------")
    print("Run Streamlit separately:\n  streamlit run text_cleaning_HW_8.py")

if __name__ == "__main__":
    main()
