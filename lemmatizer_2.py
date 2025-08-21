import subprocess, sys
from pathlib import Path

# scripts to run (console-only; Streamlit apps will be skipped automatically)
SCRIPTS = [
    "steaming_1.py",
    "lemmatizer_2.py",
    "spacy_lammatizer_3.py",
    "Project_steaming_and_lammatization_4.py",
    "bow_1.py",
    "bow_project_2.py",
    "tf-idf_1.py",
    # "text_cleaning_HW_8.py",  # streamlit -> run separately
]

ROOT = Path(__file__).parent

def run_py(script):
    path = ROOT / script
    if not path.exists():
        print(f"⚠️  SKIP (missing): {script}")
        return
    if "streamlit" in script.lower():
        print(f"⚠️  SKIP (Streamlit app): {script}")
        return
    print(f"\n========== Running {script} ==========\n")
    try:
        subprocess.run([sys.executable, str(path)], check=True)
        print(f"\n✅ Done: {script}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script}: {e}\n")

def main():
    for s in SCRIPTS:
        run_py(s)
    print("\nAll requested console scripts processed.\n")
    print("ℹ️ Run Streamlit app separately:")
    print("    streamlit run text_cleaning_HW_8.py")

if __name__ == "__main__":
    main()
