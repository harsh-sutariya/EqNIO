import os
import time
from os import path as osp

import numpy as np
import torch
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data_glob_speed import *
from transformations import *
from metric import compute_ate_rte
from model_resnet1d import *
from model_resnet1d_eq_frame import *
from model_resnet1d_eq_frame_2vec import *
from model_resnet1d_eq_frame_o2 import *
from tqdm import tqdm
import copy
import time
import wandb

_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}


def get_model(arch):
    if arch == 'resnet18_eq_frame':
        network = Eq_Motion_Model(dim_in=2, dim_out= 1, scalar_dim_in=5, pooling_dim = 1,
                               ronin_in_dim=6, ronin_out_dim=2,hidden_dim=128, scalar_hidden_dim=128, depth=1, 
                               stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch== 'resnet18_eq_frame_2vec':
        network = Eq_Motion_Model_2vec(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
                               ronin_in_dim=6, ronin_out_dim=2,hidden_dim=128, scalar_hidden_dim=128, depth=1, 
                               stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch== 'resnet18_eq_frame_o2':
        network = Eq_Motion_Model_o2(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               ronin_in_dim=6, ronin_out_dim=2,hidden_dim=64, scalar_hidden_dim=64, depth=2, 
                               stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch== 'resnet18_eq_frame_o2_3deep':
        network = Eq_Motion_Model_o2(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               ronin_in_dim=6, ronin_out_dim=2,hidden_dim=64, scalar_hidden_dim=64, depth=3, 
                               stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch== 'resnet18_eq_frame_o2_fat':
        network = Eq_Motion_Model_o2(dim_in=3, dim_out= 2, scalar_dim_in=9, pooling_dim = 1,
                               ronin_in_dim=6, ronin_out_dim=2,hidden_dim=128, scalar_hidden_dim=128, depth=1, 
                               stride = 1, padding='same', kernel =(32,1), bias=False
                            )
    elif arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network

def preprocess_eq_frame(feat):
    feat = feat.permute(0,2,1)
    ## separate the vectors and scalars from features - gyro, accel
    gyro = torch.cat((feat[..., 0].unsqueeze(-1), feat[..., 1].unsqueeze(-1)), dim=-1).unsqueeze(-2)
    accel = torch.cat((feat[..., 3].unsqueeze(-1), feat[..., 4].unsqueeze(-1)), dim=-1).unsqueeze(-2)
    vector = torch.cat((accel, gyro), dim=-2)
    scalar = torch.cat((feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1), torch.norm(accel, dim=-1), torch.norm(gyro, dim=-1), torch.einsum('... a b, ... c b -> ... c', accel, gyro)), dim=-1)
    return vector.permute(0,1,3,2), scalar, torch.cat((feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1)), dim=-1)

def preprocess_eq_o2_frame(feat):
    feat = feat.permute(0,2,1)
    ## the order is (b,n,d)
    gyro = feat[...,:3]
    accel = feat[...,-3:]
    v1 = torch.zeros((*gyro.shape[:-1],3)).to(gyro.device)
    v2 = torch.zeros((*gyro.shape[:-1],3)).to(gyro.device)
    mask = (torch.linalg.norm(gyro[...,:-1], axis=-1) == 0).to(torch.int32).to(gyro.device)
    R = torch.Tensor([[0,-1,0],[1,0,0],[0,0,1]]).to(gyro.device)
    gyro_flip = copy.deepcopy(gyro@R.T).to(gyro.device)
    gyro_flip[...,-1] = 0
    v1[mask==0] = torch.linalg.cross(gyro[mask==0], gyro_flip[mask==0])
    v2[mask==0] = torch.linalg.cross(gyro[mask==0],v1[mask==0])
    x = torch.zeros_like(gyro).to(gyro.device)
    x[...,-2] = 1
    v1[mask==1] = torch.linalg.cross(x[mask==1], gyro[mask==1])
    v2[mask==1] = torch.linalg.cross(gyro[mask==1], v1[mask==1])
    gyro_norm = torch.linalg.norm(gyro,axis=-1,keepdims=True)
    v1 = v1 * gyro_norm /torch.linalg.norm(v1, axis=-1,keepdims=True).clamp(min=1e-7)
    v2 = v2 * gyro_norm/torch.linalg.norm(v2,axis=-1,keepdims=True).clamp(min=1e-7)
    a = accel[...,:2].unsqueeze(-2)
    v1_xy = v1[...,:2].unsqueeze(-2)
    v2_xy = v2[...,:2].unsqueeze(-2)
    scalar = torch.cat((accel[...,-1].unsqueeze(-1), v1[...,-1].unsqueeze(-1), v2[...,-1].unsqueeze(-1),torch.norm(a,dim=-1),torch.norm(v1_xy,dim=-1),torch.norm(v2_xy,dim=-1),(a@v1_xy.permute(0,1,3,2)).squeeze(-1),(v1_xy@v2_xy.permute(0,1,3,2)).squeeze(-1),(a@v2_xy.permute(0,1,3,2)).squeeze(-1)),dim=-1)
    return torch.cat((a,v1_xy,v2_xy),dim=-2).permute(0,1,3,2), scalar, torch.cat((accel[...,-1].unsqueeze(-1), v1[...,-1].unsqueeze(-1), v2[...,-1].unsqueeze(-1)), dim=-1)


def run_test(network, data_loader, device, arch, eval_mode=True):
    targets_all = []
    preds_all = []
    frames_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, _, _) in enumerate(tqdm(data_loader)):
        if 'resnet18_eq_frame_o2' in arch:
            vector, scalar, orig_sca = preprocess_eq_o2_frame(feat)
            frame, pred = network(vector.float().to(device), scalar.float().to(device), orig_sca.float().to(device))
            frame = frame.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
        elif 'resnet18_eq_frame' in arch:
            vector, scalar, orig_sca = preprocess_eq_frame(feat)
            frame, pred = network(vector.float().to(device), scalar.float().to(device), orig_sca.float().to(device))
            frame = frame.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
        else:
            pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
        if 'resnet18_eq_frame' in arch:
            frames_all.append(frame)
        # if bid==10:
        #     break
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    if 'resnet18_eq_frame' in arch:
        frames_all = np.concatenate(frames_all, axis=0)
    return targets_all, preds_all, frames_all


def add_summary(writer, loss, step, mode, use_wandb=False):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)
    
    if use_wandb:
        wandb_metrics = {}
        for i in range(loss.shape[0]):
            wandb_metrics[names[i]] = loss[i]
        wandb_metrics['{}_loss/avg'.format(mode)] = np.mean(loss)
        wandb.log(wandb_metrics, step=step)


def get_dataset(root_dir, data_list, args, **kwargs):
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train' and 'resnet18_eq_frame' in args.arch:
        random_shift = args.step_size // 2
        shuffle = True
    elif mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True#False#True

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        from data_ridi import RIDIGlobSpeedSequence
        seq_type = RIDIGlobSpeedSequence
    dataset = StridedSequenceDataset(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
        random_shift=random_shift, transform=transforms,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def train(args, **kwargs):
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project="ronin", name=args.wandb_run_name if args.wandb_run_name else None, 
                  config=vars(args))
        
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    end_t = time.time()
    print('Training set loaded. Feature size: {}, target size: {}. Time usage: {:.3f}s'.format(
        train_dataset.feature_dim, train_dataset.target_dim, end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch).to(device)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    step = 0
    best_val_loss = np.inf

    print('Start from epoch {}'.format(start_epoch))
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.
    init_train_targ, init_train_pred, init_train_frames = run_test(network, train_loader, device, args.arch, eval_mode=False)

    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train', args.use_wandb)
        if args.use_wandb:
            wandb.log({'train/loss_avg': np.mean(init_train_loss)}, step=0)

    if val_loader is not None:
        init_val_targ, init_val_pred, init_val_frames = run_test(network, val_loader, device, args.arch)
        init_val_loss = np.mean((init_val_targ - init_val_pred) ** 2, axis=0)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val', args.use_wandb)
            if args.use_wandb:
                wandb.log({'val/loss_avg': np.mean(init_val_loss)}, step=0)

    try:
        for epoch in tqdm(range(start_epoch, args.epochs)):
            start_t = time.time()
            network.train()
            train_outs, train_targets = [], []
            for batch_id, (feat, targ, _, _) in enumerate(tqdm(train_loader)):
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                if 'resnet18_eq_frame_o2' in args.arch:
                    vector, scalar, orig_sca = preprocess_eq_o2_frame(feat)
                    frame, pred = network(vector.float().to(device), scalar.float().to(device), orig_sca.float().to(device))
                elif 'resnet18_eq_frame' in args.arch:
                    vector, scalar, orig_sca = preprocess_eq_frame(feat)
                    frame, pred = network(vector.float(), scalar.float(), orig_sca.float())
                else:
                    pred = network(feat)
                train_outs.append(pred.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                loss = criterion(pred, targ)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                step += 1

            train_outs = np.concatenate(train_outs, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0)

            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {}/{:.6f}'.format(
                epoch, end_t - start_t, train_losses, np.average(train_losses)))
            train_losses_all.append(np.average(train_losses))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train', args.use_wandb)
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)
                if args.use_wandb:
                    wandb.log({
                        'train/loss_avg': np.average(train_losses),
                        'optimizer/lr': optimizer.param_groups[0]['lr']
                    }, step=epoch + 1)

            if val_loader is not None:
                network.eval()
                val_outs, val_targets, val_frames = run_test(network, val_loader, device, args.arch)
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                scheduler.step(avg_loss)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val', args.use_wandb)
                    if args.use_wandb:
                        wandb.log({'val/loss_avg': avg_loss}, step=epoch + 1)
                val_losses_all.append(avg_loss)
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)
                        if args.use_wandb:
                            wandb.save(model_path)
                            wandb.run.summary['best_val_loss'] = best_val_loss
                            wandb.run.summary['best_epoch'] = epoch
            else:
                if args.out_dir is not None and osp.isdir(args.out_dir):
                    model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                    torch.save({'model_state_dict': network.state_dict(),
                                'epoch': epoch,
                                'optimizer_state_dict': optimizer.state_dict()}, model_path)
                    print('Model saved to ', model_path)
                    if args.use_wandb:
                        wandb.save(model_path)

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        print('Checkpoint saved to ', model_path)
        if args.use_wandb:
            wandb.save(model_path)
            
    # Close wandb run if it was used
    if args.use_wandb:
        wandb.finish()

    return train_losses_all, val_losses_all


def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int32)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


def test_sequence(args):
    # Initialize wandb for test mode if enabled
    if args.use_wandb:
        wandb.init(project="ronin", name=args.wandb_run_name if args.wandb_run_name else None,
                  config=vars(args))
        
    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.root_dir
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    # Load the first sequence to update the input and output size
    _ = get_dataset(root_dir, [test_data_list[0]], args)

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch)

    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []

    pred_per_min = 200 * 60

    for data in test_data_list:
        seq_dataset = get_dataset(root_dir, [data], args, mode='test')
        seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
        ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int32)

        targets, preds, frames = run_test(network, seq_loader, device, args.arch,True)
        losses = np.mean((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)

        pos_pred = recon_traj_with_preds(seq_dataset, preds)
        pos_gt = seq_dataset.gt_pos[0]

        traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
        print(pos_pred.shape,pos_gt.shape)
        ate, rte = compute_ate_rte(pos_pred[:,:2], pos_gt[:,:2], pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
        pos_cum_error = np.linalg.norm(pos_pred[:,:2] - pos_gt[:,:2], axis=1)

        print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))
        
        # Log test results to wandb if enabled
        if args.use_wandb:
            wandb_test_metrics = {
                f'test/{data}/ate': ate,
                f'test/{data}/rte': rte,
                f'test/{data}/avg_loss': np.mean(losses),
            }
            for i, loss_val in enumerate(losses):
                if i == 0:
                    wandb_test_metrics[f'test/{data}/loss_vx'] = loss_val
                elif i == 1:
                    wandb_test_metrics[f'test/{data}/loss_vy'] = loss_val
                elif i == 2:
                    wandb_test_metrics[f'test/{data}/loss_vz'] = loss_val
            wandb.log(wandb_test_metrics)

        # Plot figures
        kp = preds.shape[1]
        if kp == 2:
            targ_names = ['vx', 'vy']
        elif kp == 3:
            targ_names = ['vx', 'vy', 'vz']

        plt.figure('{}'.format(data), figsize=(16, 9))
        plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        plt.plot(pos_pred[:, 0], pos_pred[:, 1])
        plt.plot(pos_gt[:, 0], pos_gt[:, 1])
        plt.title(data)
        plt.axis('equal')
        plt.legend(['Predicted', 'Ground truth'])
        plt.subplot2grid((kp, 2), (kp - 1, 0))
        plt.plot(pos_cum_error)
        plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
        for i in range(kp):
            plt.subplot2grid((kp, 2), (i, 1))
            plt.plot(ind, preds[:, i])
            plt.plot(ind, targets[:, i])
            plt.legend(['Predicted', 'Ground truth'])
            plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
        plt.tight_layout()
        
        # Log trajectory figure to wandb if enabled
        if args.use_wandb:
            wandb.log({f"trajectory/{data}": wandb.Image(plt)})

        if args.show_plot:
            plt.show()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            if 'resnet18_eq_frame' in args.arch:
                np.save(osp.join(args.out_dir, data + '_before_traj_recon.npy'),
                    np.concatenate([preds, targets, frames.reshape((preds.shape[0], -1))], axis=1))
                np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                    np.concatenate([pos_pred, pos_gt], axis=1))
            else:
                np.save(osp.join(args.out_dir, data + '_before_traj_recon.npy'),
                    np.concatenate([preds, targets], axis=1))
                np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                        np.concatenate([pos_pred, pos_gt], axis=1))
            plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

        plt.close('all')

    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    # Export a csv file
    if args.out_dir is not None and osp.isdir(args.out_dir):
        with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
            if losses_seq.shape[1] == 2:
                f.write('seq,vx,vy,avg,ate,rte\n')
            else:
                f.write('seq,vx,vy,vz,avg,ate,rte\n')
            for i in range(losses_seq.shape[0]):
                f.write('{},'.format(test_data_list[i]))
                for j in range(losses_seq.shape[1]):
                    f.write('{:.6f},'.format(losses_seq[i][j]))
                f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

    # Log overall test results to wandb
    if args.use_wandb:
        wandb.log({
            'test/mean_ate': np.mean(ate_all),
            'test/mean_rte': np.mean(rte_all),
            'test/median_ate': np.median(ate_all),
            'test/median_rte': np.median(rte_all)
        })
        wandb.finish()

    print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}, median ATE:{}, median RTE:{}'.format(
        np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all), np.median(ate_all), np.median(rte_all)))
    return losses_avg


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, default='lists/list_train.txt')
    parser.add_argument('--val_list', type=str, default="lists/list_val.txt")
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default="output/resnet_train_cache", help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--arch', type=str, default='resnet18')#resnet18 resnet18_eq_frame_o2
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Name for the wandb run')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)#"output/ronin_original/checkpoints/checkpoint_82.pt"
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test_sequence(args)
    else:
        raise ValueError('Undefined mode')
