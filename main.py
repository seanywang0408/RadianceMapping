import os
import time
import torch
from utils import config_parser, load_fragments, load_idx, lr_decay, write_video, mse2psnr
from dataset.dataset import nerfDataset, ScanDataset, DTUDataset
from model.renderer import Renderer
import matplotlib.pyplot as plt
import torch.optim as optim
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter
from piqa import SSIM, LPIPS, PSNR
import lpips

parser = config_parser()
args = parser.parse_args()

set_seed(42)
back_path = os.path.join('./logs/', time.strftime("%y%m%d-%H%M%S-" + f'{args.expname}-{args.H}-{args.train_size}-{args.U}-{args.udim}-{args.vgg_l}-pix{args.pix_mask}-xyznear{args.xyznear}-{args.scale_min}-{args.scale_max}'))
os.makedirs(back_path)
backup_terminal_outputs(back_path)
backup_code(back_path, ignored_in_current_folder=['back','pointcloud','data','.git','pytorch_rasterizer.egg-info','build','logs','__pycache__'])
print(back_path)
logger = SummaryWriter(back_path)
video_path = os.path.join(back_path, 'video')
os.makedirs(video_path)


if __name__ == '__main__':

    if args.dataset == 'nerf':
        train_set = nerfDataset(args, 'train', 'render')
        test_set = nerfDataset(args, 'test', 'render')
    elif args.dataset == 'scan':
        train_set = ScanDataset(args, 'train', 'render')
        test_set = ScanDataset(args, 'test', 'render')
    elif args.dataset == 'dtu':
        train_set = DTUDataset(args, 'train', 'render')
        test_set = DTUDataset(args, 'test', 'render')
    else:
        assert False

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    
    renderer = Renderer(args)
    edge = args.edge_mask
    
    # Optimizer
    opt_para = []
    opt_para.append({"params": renderer.unet.parameters(), "lr": args.u_lr})  
    opt_para.append({"params": renderer.mlp.parameters(), "lr": args.mlp_lr})  
    opt = optim.Adam(opt_para)

    fn_psnr = PSNR().to(args.device)
    fn_lpips = LPIPS('vgg').to(args.device)
    loss_lpips = lpips.LPIPS(net='vgg').to(args.device)
    fn_ssim = SSIM().to(args.device)
    
    # load buf
    if args.xyznear:
        train_buf, test_buf = load_fragments(args)  # cpu 100 800 800 k
        xyz_o = None
    else:
        train_buf, test_buf = load_idx(args)  # cpu 100 800 800 k
        xyz_o = train_set.get_pc().xyz # n 3
    print('buf shape', train_buf.shape)
    
    it = 0
    epoch = 0
    best_psnr = 0
    while True:
        renderer.train()
        epoch += 1
        for batch in train_loader:
            it += 1
            idx = int(batch['idx'][0])
            ray = batch['ray'][0] # h w 7
            img_gt = batch['rgb'][0] # h w 3
            if args.dataset == 'dtu':
                mask_gt = batch['mask'][0][..., :1] # h w 1
            else:
                mask_gt = None

            zbuf = train_buf[idx].to(args.device) # h w 1

            output = renderer(zbuf, ray, img_gt, mask_gt, isTrain=True, xyz_o=xyz_o)

            if args.dataset == 'dtu':
                img_pre = output['img'] * output['mask_gt'] + 1 - output['mask_gt']
            else:
                img_pre = output['img']

            if output['gt'].min() == 1:
                print('None img, skip')
                # torch.cuda.empty_cache()
                continue

            opt.zero_grad()
            # if edge > 0:
            #     loss_l2 = torch.mean((img_pre - output['gt'])[edge:-edge, edge:-edge] ** 2)
            # else:
            loss_l2 = torch.mean((img_pre - output['gt']) ** 2)

            if args.vgg_l > 0:
                loss_vgg = loss_lpips(img_pre.permute(2,0,1).unsqueeze(0), output['gt'].permute(2,0,1).unsqueeze(0), normalize=True)
                loss = loss_l2 + args.vgg_l * loss_vgg
            else:
                loss = loss_l2 

            loss.backward()
            opt.step()

            if it % 50 == 0:
                psnr = mse2psnr(loss_l2)
                logger.add_scalar('train/psnr', psnr.item(), it)

            if it % 200 == 0:
                if args.vgg_l > 0:
                    print('[{}]-it:{}, psnr:{:.4f}, l2_loss:{:.4f}, vgg_loss:{:.4f}'.format(back_path, it, psnr.item(), loss_l2.item(), loss_vgg.item()))
                else:
                    print('[{}]-it:{}, psnr:{:.4f}, l2_loss:{:.4f}'.format(back_path, it, psnr.item(), loss.item()))
                img_pre[img_pre>1] = 1.
                img_pre[img_pre<0] = 0.
                logger.add_image('train/fea', output['fea_map'], global_step=it, dataformats='HWC')
                logger.add_image('train/img_fine', img_pre, global_step=it, dataformats='HWC')
                logger.add_image('train/gtimg', output['gt'], global_step=it, dataformats='HWC')
            torch.cuda.empty_cache()
        lr_decay(opt)


        # test
        if epoch % args.test_freq == 0:
            print('TEST BEGIN!!!')
            if epoch % args.vid_freq == 0:
                video_it_path = os.path.join(video_path, str(it))
                os.makedirs(video_it_path)

            test_psnr = 0
            test_lpips = 0
            test_ssim = 0

            renderer.eval()
            with torch.autograd.no_grad():
                for i, batch in enumerate(test_loader):
                    idx = int(batch['idx'][0])
                    ray = batch['ray'][0]
                    img_gt = batch['rgb'][0]
                    zbuf = test_buf[idx].to(args.device)
                    output = renderer(zbuf, ray, gt=None, mask_gt=None, isTrain=False, xyz_o=xyz_o)

                    if args.dataset == 'dtu':
                        mask_gt = batch['mask'][0][..., :1]
                        img_pre = output['img'].detach()[...,:3] * mask_gt + 1 - mask_gt
                    else:
                        img_pre = output['img']

                    img_pre[img_pre>1] = 1.
                    img_pre[img_pre<0] = 0.

                    img_pre = img_pre.permute(2,0,1).unsqueeze(0)
                    img_gt = img_gt.permute(2,0,1).unsqueeze(0)


                    if edge > 0:
                        psnr = fn_psnr(img_pre[...,edge:-edge,edge:-edge], img_gt[...,edge:-edge,edge:-edge])
                        ssim = fn_ssim(img_pre[...,edge:-edge,edge:-edge], img_gt[...,edge:-edge,edge:-edge])
                        lpips_ = fn_lpips(img_pre[...,edge:-edge,edge:-edge], img_gt[...,edge:-edge,edge:-edge])
                    else:
                        psnr = fn_psnr(img_pre, img_gt)
                        ssim = fn_ssim(img_pre, img_gt)
                        lpips_ = fn_lpips(img_pre, img_gt)
                    test_lpips += lpips_.item()
                    test_psnr += psnr.item()
                    test_ssim += ssim.item()

                    if epoch % args.vid_freq == 0: 

                        img_pre = img_pre.squeeze(0).permute(1,2,0)
                        img_pre = img_pre.cpu().numpy()
                        plt.imsave(os.path.join(video_it_path, str(i).rjust(3,'0') + '.png'), img_pre)
                    # torch.cuda.empty_cache()
                    
            test_lpips = test_lpips / len(test_set)
            test_psnr = test_psnr / len(test_set)
            test_ssim = test_ssim / len(test_set)
            logger.add_scalar('test/psnr', test_psnr, it)
            logger.add_scalar('test/lpips', test_lpips, it)
            logger.add_scalar('test/ssim', test_ssim, it)

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                ckpt = os.path.join(back_path, 'model.pkl')
                torch.save(renderer.state_dict(), ckpt)
                print('model saved!!!!!!!!', best_psnr)


            # write_video(video_path, os.path.join(video_path, 'video.avi'), (args.W, args.H))
            print('test set psnr!!!', test_psnr, best_psnr)
            # print('write video!!!!!!!!!')