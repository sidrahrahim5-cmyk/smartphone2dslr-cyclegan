import os

class Config:
    # ─── Dataset ───────────────────────────────────────
    DATASET_NAME    = "smartphone2dslr_flowers"
    DATA_ROOT       = "./dataset"
    DOMAIN_A        = "trainA"   # Smartphone photos
    DOMAIN_B        = "trainB"   # DSLR photos

    # ─── Image ─────────────────────────────────────────
    IMG_SIZE        = 256
    IMG_CHANNELS    = 3

    # ─── Training ──────────────────────────────────────
    EPOCHS          = 200
    DECAY_EPOCH     = 100        # LR linear decay start
    BATCH_SIZE      = 1          # Paper: batch=1
    LR              = 0.0002     # Paper: Adam lr
    BETA1           = 0.5
    BETA2           = 0.999
    LAMBDA_CYCLE    = 10.0       # Paper: λ = 10
    LAMBDA_IDENTITY = 5.0        # 0.5 × λ

    # ─── Model ─────────────────────────────────────────
    N_RESIDUAL_BLOCKS = 9        # 256×256 images ke liye

    # ─── Replay Buffer ─────────────────────────────────
    BUFFER_SIZE     = 50         # Paper: 50 previously generated images

    # ─── Paths ─────────────────────────────────────────
    CHECKPOINT_DIR  = "./checkpoints"
    RESULTS_DIR     = "./assets/results"
    LOG_EVERY       = 100        # batches

    # ─── Device ────────────────────────────────────────
    DEVICE          = "cuda"     # Colab GPU

    @classmethod
    def make_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.DATA_ROOT, exist_ok=True)
        print("✅ Directories ready!")