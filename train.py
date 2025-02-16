import click
import os
import torch
from torch.cuda.amp import autocast, GradScaler
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import Plants, collate_pdc
import models
import yaml
import time
from tqdm import tqdm

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True


def save_model(model, epoch, optim, scheduler, n_iter, best_map_det, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'n_iter': n_iter,
        'best_map_det': best_map_det,
        }, name)

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/cfg.yaml'))
@click.option('--percentage',
              '-p',
              type=float,
              help='percentage of training data to be used for training, if train_list specified in the cfg file',
              default=1.0)
@click.option('--resume',
              '-r',
              type=str,
              help='path to checkpoint file to resume training from',
              default=None)
def main(config, percentage, resume):
    cfg = yaml.safe_load(open(config))
    cfg['data']['percentage'] = percentage

    train_dataset = Plants(datapath=cfg['data']['train'], overfit=cfg['train']['overfit'], cfg=cfg, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'],
                              collate_fn=collate_pdc, shuffle=True, drop_last=True, 
                              num_workers=cfg['train']['workers'], pin_memory=True,
                              persistent_workers=True)
    if cfg['train']['overfit']:
        val_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'],
                                collate_fn=collate_pdc, shuffle=False, drop_last=False, num_workers=cfg['train']['workers'])
    else:
        val_dataset = Plants(datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
        val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False, drop_last=False, num_workers=cfg['train']['workers'])

    model = models.get_model(cfg)
    model = model.cuda()  # Move model to GPU
    if torch.__version__ >= '2.0.0':
        model = torch.compile(model, mode='max-autotune')  # Enable torch.compile with max autotune
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Get gradient accumulation steps from config
    grad_accum_steps = cfg['train'].get('gradient_accumulation_steps', 1)
    effective_batch_size = cfg['train']['batch_size'] * grad_accum_steps
    print(f'Training with effective batch size: {effective_batch_size} (batch_size: {cfg["train"]["batch_size"]} x {grad_accum_steps} accumulation steps)')
    optim = torch.optim.AdamW(model.network.parameters(), lr=cfg['train']['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

    # Initialize training state
    start_epoch = 0
    n_iter = 0
    best_map_det = 0

    # Load checkpoint if resuming
    if resume is not None and os.path.exists(resume):
        print(f'Loading checkpoint from {resume}')
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        n_iter = checkpoint['n_iter']
        best_map_det = checkpoint['best_map_det']
        print(f'Resuming from epoch {start_epoch} (iteration {n_iter})')

    with torch.autograd.set_detect_anomaly(True):
        for e in range(start_epoch, cfg['train']['max_epoch']):
            model.network.train()
            start = time.time()
            optim.zero_grad()  # Zero gradients at the start of each epoch
            for idx, item in enumerate(iter(train_loader)):
                # Mixed precision training with gradient accumulation
                with autocast():
                    loss = model.training_step(item)
                    # Normalize loss by gradient accumulation steps
                    loss = loss / grad_accum_steps
                
                # Scale loss and backward pass
                scaler.scale(loss).backward()
                
                # Step optimization every grad_accum_steps or at the end of epoch
                if ((idx + 1) % grad_accum_steps == 0) or (idx == len(train_loader) - 1):
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                
                it_time = time.time() - start
                print('Epoch: {}/{} -- Step: {}/{} -- Loss: {} -- Lr: {} -- Time: {} -- Acc.Step: {}/{}'.format(
                    e, cfg['train']['max_epoch'], idx*cfg['train']['batch_size'], len(train_dataset), 
                    loss.item() * grad_accum_steps,  scheduler.get_lr()[0], it_time,
                    (idx % grad_accum_steps) + 1, grad_accum_steps
                ))
                model.writer.add_scalar('Loss/Train/', loss.detach().cpu().item(), n_iter)
                n_iter += 1
                start = time.time()
            
            scheduler.step()
            name = os.path.join(model.ckpt_dir, 'last.pt')
            save_model(model, e, optim, scheduler, n_iter, best_map_det, name)
            
            model.network.eval()
            model.on_validation_start()
            for idx, item in enumerate(tqdm(val_loader)):
                with torch.no_grad():
                    model.validation_step(item)

            ap_detection = model.compute_metrics()
            model.writer.add_scalar('Metric/Val/mAP_detection', ap_detection['map'].item(), n_iter)

            if model.log_val_predictions:
                bbs = model.img_with_box[:4]
                for batch_idx in range(len(bbs)):
                    model.writer.add_image("Boxes/" + "b" + str(batch_idx), bbs[batch_idx], n_iter, dataformats='HWC')

            # checking improvements on validation set
            if ap_detection['map'].item() >= best_map_det:
                name = os.path.join(model.ckpt_dir, 'best_detection_map.pt')
                save_model(model, e, optim, scheduler, n_iter, best_map_det, name)
        
if __name__ == "__main__":
    main()