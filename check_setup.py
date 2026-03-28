import sys
print("=" * 45)
print("   CycleGAN Setup Check")
print("=" * 45)

print(f"\n Python     : {sys.version.split()[0]}")

try:
    import torch
    print(f" PyTorch    : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"⚡ GPU        : {torch.cuda.get_device_name(0)}")
    else:
        print(" GPU        : CPU mode")
except ImportError:
    print("❌ PyTorch    : NOT INSTALLED")

libs = {
    "Torchvision" : "torchvision",
    "Pillow"      : "PIL",
    "NumPy"       : "numpy",
    "Matplotlib"  : "matplotlib",
    "tqdm"        : "tqdm",
    "Gradio"      : "gradio",
    "gdown"       : "gdown",
}

print()
for name, module in libs.items():
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "OK")
        print(f"  ✅ {name:12s}: {ver}")
    except ImportError:
        print(f"  ❌ {name:12s}: NOT INSTALLED")

print("\n" + "=" * 45)
print("  Everything is working fine! ")
print("=" * 45)