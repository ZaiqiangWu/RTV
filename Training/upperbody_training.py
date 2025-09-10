import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))
from collections import OrderedDict


from Datasets.upperbody_garment.upperbody_garment import UpperBodyGarment
from options.train_options import TrainOptions
from model.pix2pixHD.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torchvision
import torch
import math
def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0
import time

def main():
    opt = TrainOptions().parse()
    if opt.dataset_path is not None:
        dataset_paths = opt.dataset_path
        path_list = dataset_paths.split(',')
        dataset = UpperBodyGarment(path_list[0],img_size=opt.img_size)
        if len(path_list) > 1:
            for i in range(1,len(path_list)):
                dataset = dataset + UpperBodyGarment(path_list[i], img_size=opt.img_size)
    else:
        print("Please specify a dataset for training!")
        exit(0)
    dataset_size=len(dataset)
    dataloader=torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
    model=create_model(opt)
    visualizer = Visualizer(opt)
    start_epoch, epoch_iter = 1, 0
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataloader):
            # forward
            garment_img, vm_img, dp_img, garment_mask = data
            #pred_mask = model.forward(dp, garment_mask)

            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            #losses, generated = model.module.forward_attention(vm_img, garment_img,attrn_mask, infer=save_fake)
            input_img = torch.cat([vm_img,dp_img],1)
            gt_image=torch.cat([garment_img,garment_mask],1)
            losses, generated = model(input_img, gt_image, infer=save_fake)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            ############### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()

            loss_G.backward()
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()

            loss_D.backward()
            optimizer_D.step()

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
                # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

            ### display output images
            if save_fake:
                real_list = [('garment_img' + str(k), util.tensor2im((garment_img/2.0+0.5)[0],rgb=True)) for k in range(1)]
                fake_list = [('fake_img' + str(k), util.tensor2im((generated.data[:,[0,1,2],:,:] / 2.0 + 0.5)[0], rgb=True)) for k in
                             range(1)]
                fake2_list = [
                    ('fake_mask' + str(k), util.tensor2im((generated.data[:, [3,3,3], :, :])[0], rgb=True))
                    for k in
                    range(1)]
                input_list = [('vm_image' + str(k), util.tensor2im((vm_img/2.0+0.5)[0],rgb=True)) for k in range(1)]
                dp_list = [('dp_image' + str(k), util.tensor2im((dp_img / 2.0 + 0.5)[0], rgb=True)) for k in
                              range(1)]
                visuals = OrderedDict( real_list + input_list+fake_list+dp_list+fake2_list)
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')



            if epoch_iter >= dataset_size:
                break

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()



if __name__=="__main__":
    main()