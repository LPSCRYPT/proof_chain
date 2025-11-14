from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Training Hyperparameters
num_epochs = 50
fixed_noise = torch.randn(num_classes, latent_dim, 1, 1, device=device)
fixed_labels = torch.arange(0, num_classes, dtype=torch.long, device=device)

print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.size(0)

        # --- Train Discriminator ---
        discriminator.zero_grad()

        # 1. Train with real images (with label smoothing)
        # Use a value slightly less than 1, e.g., random between 0.8 and 1.0
        real_labels_d = (torch.rand(batch_size, 1, 1, 1, device=device) * 0.2 + 0.8) # Label '1' for real images
        output = discriminator(imgs, labels)
        errD_real = criterion(output, real_labels_d)
        errD_real.backward()

        # 2. Train with fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=device) # Random labels for fake images
        fake_imgs = generator(noise, fake_labels) # Generate fake images

        fake_labels_d = torch.zeros(batch_size, 1, 1, 1, device=device) # Label '0' for fake images
        output = discriminator(fake_imgs.detach(), fake_labels) # Detach fake_imgs from generator history
        errD_fake = criterion(output, fake_labels_d)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizer_d.step()

        # --- Train Generator (twice for every one Discriminator update) ---
        num_generator_updates = 2
        for _ in range(num_generator_updates):
            generator.zero_grad()

            # Generate new noise and fake labels for each generator update
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_imgs = generator(noise, fake_labels)

            # Since we want the generator to fool the discriminator, we use smoothed real_labels_d here
            output = discriminator(fake_imgs, fake_labels)
            errG = criterion(output, real_labels_d) # Generator wants D to classify fakes as real (label '1')
            errG.backward()
            optimizer_g.step()

        # --- Print Progress ---
        if i % 100 == 0 or i == len(trainloader) - 1:
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(trainloader)}] "
                       f"Loss D: {errD.item():.4f} Loss G: {errG.item():.4f}")

    # --- Save generated images for visualization after each epoch ---
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        generator.eval() # Set generator to evaluation mode
        with torch.no_grad():
            generated_samples = generator(fixed_noise, fixed_labels).detach().cpu()

            # Unnormalize images for better visualization
            # CIFAR-10 normalization values: mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
            generated_samples = generated_samples * std + mean # Denormalize
            generated_samples = torch.clamp(generated_samples, 0, 1) # Clamp to [0, 1] range

            fig, axes = plt.subplots(1, num_classes, figsize=(20, 2))
            for j in range(num_classes):
                ax = axes[j]
                img = np.transpose(generated_samples[j].numpy(), (1, 2, 0))
                ax.imshow(img)
                ax.set_title(f"Class {j}")
                ax.axis('off')
            plt.suptitle(f"Generated Images after Epoch {epoch+1}", fontsize=16)
            plt.show()
        generator.train() # Set generator back to training mode

print("Training complete.")