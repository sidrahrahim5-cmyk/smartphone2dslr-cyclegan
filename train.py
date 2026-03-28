import os
import time
import torch
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.models import Generator, Discriminator, GANLoss, CycleLoss, IdentityLoss
from src.data import SmartphoneDSLRDataset
from src.utils import ReplayBuffer, save_sample_grid


def weights_init(m):
    """
    Initialize network weights from Gaussian distribution.
    Paper: weights initialized from N(0, 0.02)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_lr_lambda(config):
    """
    Linear learning rate decay.
    Keep LR constant for first DECAY_EPOCH epochs,
    then linearly decay to 0 over remaining epochs.
    Paper Section 4: same strategy used.
    """
    def lr_lambda(epoch):
        if epoch < config.DECAY_EPOCH:
            return 1.0
        return 1.0 - (epoch - config.DECAY_EPOCH) / (
            config.EPOCHS - config.DECAY_EPOCH
        )
    return lr_lambda


def train():
    # ── Setup ────────────────────────────────────────────────
    config = Config()
    config.make_dirs()
    device = torch.device(config.DEVICE
                          if torch.cuda.is_available()
                          else "cpu")
    print(f"[Info] Using device: {device}")

    # ── Dataset & DataLoader ─────────────────────────────────
    dataset = SmartphoneDSLRDataset(
        root     = config.DATA_ROOT,
        mode     = "train",
        img_size = config.IMG_SIZE,
    )
    dataloader = DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle    = True,
        num_workers= 0,       # Windows compatibility
        pin_memory = True if device.type == "cuda" else False,
    )
    print(f"[Info] Batches per epoch: {len(dataloader)}")

    # ── Models ───────────────────────────────────────────────
    # G: Smartphone -> DSLR
    # F: DSLR -> Smartphone
    G = Generator(config.IMG_CHANNELS,
                  config.N_RESIDUAL_BLOCKS).to(device)
    F = Generator(config.IMG_CHANNELS,
                  config.N_RESIDUAL_BLOCKS).to(device)

    # DY: discriminates real vs fake DSLR
    DY = Discriminator(config.IMG_CHANNELS).to(device)
    # DX: discriminates real vs fake Smartphone
    DX = Discriminator(config.IMG_CHANNELS).to(device)

    # Apply weight initialization
    G.apply(weights_init)
    F.apply(weights_init)
    DY.apply(weights_init)
    DX.apply(weights_init)

    print("[Info] Models initialized with N(0, 0.02) weights")

    # ── Loss Functions ───────────────────────────────────────
    criterion_GAN      = GANLoss().to(device)
    criterion_cycle    = CycleLoss().to(device)
    criterion_identity = IdentityLoss().to(device)

    # ── Optimizers ───────────────────────────────────────────
    # Generators share one optimizer (paper trains them jointly)
    optimizer_G = torch.optim.Adam(
        itertools.chain(G.parameters(), F.parameters()),
        lr    = config.LR,
        betas = (config.BETA1, config.BETA2),
    )
    optimizer_DY = torch.optim.Adam(
        DY.parameters(),
        lr    = config.LR,
        betas = (config.BETA1, config.BETA2),
    )
    optimizer_DX = torch.optim.Adam(
        DX.parameters(),
        lr    = config.LR,
        betas = (config.BETA1, config.BETA2),
    )

    # ── LR Schedulers ────────────────────────────────────────
    lr_lambda = get_lr_lambda(config)
    scheduler_G  = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,  lr_lambda=lr_lambda)
    scheduler_DY = torch.optim.lr_scheduler.LambdaLR(
        optimizer_DY, lr_lambda=lr_lambda)
    scheduler_DX = torch.optim.lr_scheduler.LambdaLR(
        optimizer_DX, lr_lambda=lr_lambda)

    # ── Replay Buffers ───────────────────────────────────────
    # Paper: keep history of 50 generated images
    buffer_fake_B = ReplayBuffer(config.BUFFER_SIZE)
    buffer_fake_A = ReplayBuffer(config.BUFFER_SIZE)

    # ── Resume from checkpoint if exists ─────────────────────
    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(checkpoint_path):
        print(f"[Info] Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        F.load_state_dict(ckpt["F"])
        DY.load_state_dict(ckpt["DY"])
        DX.load_state_dict(ckpt["DX"])
        optimizer_G.load_state_dict(ckpt["optimizer_G"])
        optimizer_DY.load_state_dict(ckpt["optimizer_DY"])
        optimizer_DX.load_state_dict(ckpt["optimizer_DX"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Info] Resuming from epoch {start_epoch}")

    # ── Training Loop ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Starting CycleGAN Training")
    print(f"  Epochs        : {config.EPOCHS}")
    print(f"  Batch size    : {config.BATCH_SIZE}")
    print(f"  Lambda cycle  : {config.LAMBDA_CYCLE}")
    print(f"  Lambda identity: {config.LAMBDA_IDENTITY}")
    print("=" * 55 + "\n")

    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()

        # Loss accumulators for logging
        loss_G_total  = 0.0
        loss_DY_total = 0.0
        loss_DX_total = 0.0

        pbar = tqdm(dataloader,
                    desc=f"Epoch [{epoch+1}/{config.EPOCHS}]",
                    leave=True)

        for batch_idx, batch in enumerate(pbar):
            real_A = batch["A"].to(device)   # Smartphone image
            real_B = batch["B"].to(device)   # DSLR image

            # ══════════════════════════════════════════════
            #  Train Generators G and F
            # ══════════════════════════════════════════════
            optimizer_G.zero_grad()

            # --- Identity loss ---
            # G(real_B) should equal real_B (already DSLR)
            # F(real_A) should equal real_A (already Smartphone)
            same_B = G(real_B)
            same_A = F(real_A)
            loss_identity = (
                criterion_identity(same_B, real_B) +
                criterion_identity(same_A, real_A)
            ) * config.LAMBDA_IDENTITY

            # --- GAN loss ---
            # G tries to fool DY: DY(G(A)) should be 1 (real)
            # F tries to fool DX: DX(F(B)) should be 1 (real)
            fake_B = G(real_A)
            fake_A = F(real_B)
            loss_GAN = (
                criterion_GAN(DY(fake_B), is_real=True) +
                criterion_GAN(DX(fake_A), is_real=True)
            )

            # --- Cycle consistency loss ---
            # Forward:  A -> G(A) -> F(G(A)) ~ A
            # Backward: B -> F(B) -> G(F(B)) ~ B
            rec_A = F(fake_B)
            rec_B = G(fake_A)
            loss_cycle = (
                criterion_cycle(rec_A, real_A) +
                criterion_cycle(rec_B, real_B)
            ) * config.LAMBDA_CYCLE

            # --- Total generator loss ---
            loss_G = loss_GAN + loss_cycle + loss_identity
            loss_G.backward()
            optimizer_G.step()

            # ══════════════════════════════════════════════
            #  Train Discriminator DY (real vs fake DSLR)
            # ══════════════════════════════════════════════
            optimizer_DY.zero_grad()

            # Use replay buffer to get historical fake images
            fake_B_buffered = buffer_fake_B.push_and_pop(fake_B.detach())

            loss_DY = (
                criterion_GAN(DY(real_B),          is_real=True)  +
                criterion_GAN(DY(fake_B_buffered), is_real=False)
            ) * 0.5      # Paper: divide by 2 to slow D learning

            loss_DY.backward()
            optimizer_DY.step()

            # ══════════════════════════════════════════════
            #  Train Discriminator DX (real vs fake Phone)
            # ══════════════════════════════════════════════
            optimizer_DX.zero_grad()

            fake_A_buffered = buffer_fake_A.push_and_pop(fake_A.detach())

            loss_DX = (
                criterion_GAN(DX(real_A),          is_real=True)  +
                criterion_GAN(DX(fake_A_buffered), is_real=False)
            ) * 0.5

            loss_DX.backward()
            optimizer_DX.step()

            # ── Accumulate losses ─────────────────────────
            loss_G_total  += loss_G.item()
            loss_DY_total += loss_DY.item()
            loss_DX_total += loss_DX.item()

            # ── Progress bar update ───────────────────────
            pbar.set_postfix({
                "G"  : f"{loss_G.item():.3f}",
                "DY" : f"{loss_DY.item():.3f}",
                "DX" : f"{loss_DX.item():.3f}",
            })

            # ── Save sample images every N batches ───────
            if batch_idx % config.LOG_EVERY == 0:
                save_sample_grid(
                    real_A, fake_B, rec_A,
                    real_B, fake_A, rec_B,
                    save_dir  = config.RESULTS_DIR,
                    epoch     = epoch + 1,
                    batch_idx = batch_idx,
                )

        # ── End of epoch ─────────────────────────────────
        n_batches = len(dataloader)
        epoch_time = time.time() - epoch_start

        print(f"\n[Epoch {epoch+1:03d}] "
              f"G: {loss_G_total/n_batches:.4f} | "
              f"DY: {loss_DY_total/n_batches:.4f} | "
              f"DX: {loss_DX_total/n_batches:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # ── Update learning rate schedulers ──────────────
        scheduler_G.step()
        scheduler_DY.step()
        scheduler_DX.step()
        print(f"[LR] {scheduler_G.get_last_lr()[0]:.6f}")

        # ── Save checkpoint every 10 epochs ──────────────
        if (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch"        : epoch,
                "G"            : G.state_dict(),
                "F"            : F.state_dict(),
                "DY"           : DY.state_dict(),
                "DX"           : DX.state_dict(),
                "optimizer_G"  : optimizer_G.state_dict(),
                "optimizer_DY" : optimizer_DY.state_dict(),
                "optimizer_DX" : optimizer_DX.state_dict(),
            }
            # Latest checkpoint (overwrite)
            torch.save(ckpt, checkpoint_path)

            # Epoch-specific checkpoint
            epoch_ckpt = os.path.join(
                config.CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pth"
            )
            torch.save(ckpt, epoch_ckpt)
            print(f"[Saved] Checkpoint: {epoch_ckpt}")

    print("\n Training Complete!")
    print(f" Results saved in: {config.RESULTS_DIR}")
    print(f" Checkpoints saved in: {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()