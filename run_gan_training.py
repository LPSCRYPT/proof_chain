#!/usr/bin/env python3
import sys
import torch
from train_zk_conditional_gan import AntiCollapseTrainer
from zk_optimized_conditional_gan import ZKOptimizedGeneratorV2, ZKOptimizedDiscriminator

def main():
    print('='*60)
    print('STARTING ZK-OPTIMIZED CONDITIONAL GAN TRAINING')
    print('='*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize Balanced (V2) architecture - best trade-off
    print('\nInitializing Balanced (V2) architecture...')
    generator = ZKOptimizedGeneratorV2(
        latent_dim=100, 
        num_classes=10, 
        embed_dim=50, 
        ngf=48
    ).to(device)
    
    discriminator = ZKOptimizedDiscriminator(
        num_classes=10,
        ndf=64
    ).to(device)
    
    print(f'Generator parameters: {sum(p.numel() for p in generator.parameters()):,}')
    print(f'Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}')
    
    # Initialize trainer with anti-collapse measures
    print('\nSetting up anti-collapse trainer...')
    trainer = AntiCollapseTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device
    )
    
    # Setup optimizers with conservative learning rates
    trainer.setup_optimizers(lr_g=0.0001, lr_d=0.0002)
    
    # Start training
    print('\nStarting training for 100 epochs...')
    print('Anti-collapse measures enabled:')
    print('  - Label smoothing: 0.1')
    print('  - Instance noise: 0.1 (decaying)')
    print('  - Gradient clipping: 5.0')
    print('  - Auto-recovery on collapse')
    print('  - Diversity regularization')
    print()
    
    try:
        loss_history, quality_history = trainer.train(num_epochs=100)
        
        print('\n' + '='*60)
        print('TRAINING COMPLETE')
        print('='*60)
        
        # Save final model
        print('Saving final models...')
        torch.save(generator.state_dict(), 'cifar_gan_training/zk_conditional_gan_v2_final.pth')
        torch.save(discriminator.state_dict(), 'cifar_gan_training/zk_discriminator_v2_final.pth')
        print('✓ Models saved')
        
        # Show final metrics
        if quality_history:
            last_metrics = quality_history[-1]
            print(f'\nFinal metrics:')
            print(f'  Diversity: {last_metrics.get("diversity", 0):.4f}')
            print(f'  Entropy: {last_metrics.get("entropy", 0):.4f}')
            
    except KeyboardInterrupt:
        print('\n⚠️ Training interrupted by user')
        print('Saving checkpoint...')
        torch.save(generator.state_dict(), 'cifar_gan_training/zk_conditional_gan_v2_checkpoint.pth')
        torch.save(discriminator.state_dict(), 'cifar_gan_training/zk_discriminator_v2_checkpoint.pth')
        print('✓ Checkpoint saved')
    
    except Exception as e:
        print(f'\n✗ Training failed: {e}')
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
