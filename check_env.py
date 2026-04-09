import subprocess
import sys
import importlib.util

def check_command(cmd, name):
    try:
        subprocess.run([cmd, "--version"], capture_output=True, text=True, check=True)
        print(f"✅ {name} runs successfully.")
        return True
    except FileNotFoundError:
        print(f"❌ {name} not found in PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} found but returned an error: {e}")
        return False

def check_package(pkg_name):
    if importlib.util.find_spec(pkg_name) is not None:
        print(f"✅ Python package '{pkg_name}' found.")
        return True
    else:
        print(f"❌ Python package '{pkg_name}' is missing.")
        return False

def main():
    print("--- Environment Check for Audio Stem Splitter ---")
    all_good = True
    
    # Check CLI tools
    if not check_command("ffmpeg", "FFmpeg"):
        all_good = False
    
    if not check_command("yt-dlp", "yt-dlp"):
        all_good = False

    # Check basic python packages
    packages_to_check = [
        "mcp",
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "demucs",
        "psutil"
    ]
    for pkg in packages_to_check:
        if not check_package(pkg):
            all_good = False
            
    print("-------------------------------------------------")
    if all_good:
        print("🎉 System looks ready to run the Audio Stem Splitter server.")
        sys.exit(0)
    else:
        print("⚠️  There are missing dependencies. Please check requirements.txt and your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    main()
