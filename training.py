
global_step = 0
for epoch in range(1, NUM_EPOCHS+1):
    G.train(); D1.train(); D2.train()
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0
    epoch_l1 = 0.0
    epoch_ssim = 0.0
    epoch_psnr = 0.0
    n_batches = 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
    for i, (src, tgt) in enumerate(loop):
        src = src.to(device); tgt = tgt.to(device)
        batch_size = src.size(0)
        n_batches += 1
        global_step += 1

        fake = G(src)

        # D1 update
        optimizer_D1.zero_grad()
        real_input = torch.cat([src, tgt], dim=1)
        fake_input = torch.cat([src, fake.detach()], dim=1)
        out_real_D1 = D1(real_input); out_fake_D1 = D1(fake_input)
        real_label = torch.ones_like(out_real_D1, device=device); fake_label = torch.zeros_like(out_fake_D1, device=device)
        loss_D1 = 0.5 * (criterion_GAN(out_real_D1, real_label) + criterion_GAN(out_fake_D1, fake_label))
        loss_D1.backward(); optimizer_D1.step()

        # D2 update
        optimizer_D2.zero_grad()
        out_real_D2 = D2(real_input); out_fake_D2 = D2(fake_input)
        real_label2 = torch.ones_like(out_real_D2, device=device); fake_label2 = torch.zeros_like(out_fake_D2, device=device)
        loss_D2 = 0.5 * (criterion_GAN(out_real_D2, real_label2) + criterion_GAN(out_fake_D2, fake_label2))
        loss_D2.backward(); optimizer_D2.step()

        loss_D_total = 0.5 * (loss_D1 + loss_D2)

        # Generator update
        optimizer_G.zero_grad()
        fake_input_for_G = torch.cat([src, fake], dim=1)
        pred_D1 = D1(fake_input_for_G); pred_D2 = D2(fake_input_for_G)
        target_real_for_G1 = torch.ones_like(pred_D1, device=device); target_real_for_G2 = torch.ones_like(pred_D2, device=device)
        loss_G_gan_D1 = criterion_GAN(pred_D1, target_real_for_G1)
        loss_G_gan_D2 = criterion_GAN(pred_D2, target_real_for_G2)
        loss_G_gan = 0.5 * (loss_G_gan_D1 + loss_G_gan_D2)
        loss_G_l1 = criterion_L1(fake, tgt) * L1_LAMBDA
        loss_G = loss_G_gan + loss_G_l1
        loss_G.backward(); optimizer_G.step()

        # Metrics
        ssim_b, psnr_b = compute_ssim_psnr_batch(fake, tgt)

        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D_total.item()
        epoch_l1 += loss_G_l1.item()
        epoch_ssim += ssim_b
        epoch_psnr += psnr_b

        loop.set_postfix({
            'G_total': f"{loss_G.item():.4f}",
            'D_total': f"{loss_D_total.item():.4f}",
            'L1': f"{loss_G_l1.item():.4f}",
            'SSIM': f"{ssim_b:.4f}",
            'PSNR': f"{psnr_b:.2f}"
        })

        if global_step % 100 == 0:
            save_samples(epoch, src, fake, tgt, out_dir=SAMPLES_DIR, nrow=src.size(0))


    n = max(1, n_batches)
    epoch_G_loss_avg = epoch_G_loss / n
    epoch_D_loss_avg = epoch_D_loss / n
    epoch_l1_avg = epoch_l1 / n
    epoch_ssim_avg = epoch_ssim / n
    epoch_psnr_avg = epoch_psnr / n

    metrics_row = {
        'epoch': epoch,
        'G_total': epoch_G_loss_avg,
        'G_gan_D1': None,
        'G_gan_D2': None,
        'L1': epoch_l1_avg,
        'D_total': epoch_D_loss_avg,
        'D1_loss': None,
        'D2_loss': None,
        'SSIM': epoch_ssim_avg,
        'PSNR': epoch_psnr_avg
    }
    metrics_log.append(metrics_row)
    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(metrics_csv_path, index=False)


    with torch.no_grad():
        save_samples(epoch, src, fake, tgt, out_dir=SAMPLES_DIR, nrow=src.size(0))

    ckpt = {
        'epoch': epoch,
        'G_state': G.state_dict(),
        'D1_state': D1.state_dict(),
        'D2_state': D2.state_dict(),
        'opt_G': optimizer_G.state_dict(),
        'opt_D1': optimizer_D1.state_dict(),
        'opt_D2': optimizer_D2.state_dict()
    }
    torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pth"))
    if epoch_l1_avg < best_val_l1:
        best_val_l1 = epoch_l1_avg
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"best_checkpoint_epoch_{epoch:04d}.pth"))
        print(f"Saved new best model by L1: epoch {epoch}, L1 {best_val_l1:.4f}")

    plot_and_save(df_metrics, out_dir=OUT_DIR)
    print(f"Epoch {epoch:03d} summary: G_loss={epoch_G_loss_avg:.4f}, D_loss={epoch_D_loss_avg:.4f}, L1={epoch_l1_avg:.4f}, SSIM={epoch_ssim_avg:.4f}, PSNR={epoch_psnr_avg:.2f} dB")

print("Training finished. Metrics saved to:", metrics_csv_path)
print("Checkpoints saved to:", CHECKPOINT_DIR)
print("Sample images saved to:", SAMPLES_DIR)
