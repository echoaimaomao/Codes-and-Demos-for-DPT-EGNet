import torch
import torch.nn as nn
from metric import get_stoi, get_pesq
from scipy.io import wavfile
import numpy as np
from checkpoints import Checkpoint
from torch.utils.data import DataLoader
from helper_funcs import snr, numParams
from eval_composite import eval_composite
from AudioData import TestDataset, TestCollate
# from new_model import Net
from new_model_dual_split_after_dual import Net
import h5py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sr = 16000

test_file_list_path = './female2_test_file_list'
clean_file_path = './datasets/testset_clean'
noisy_file_path = './datasets/testset_noisy'
audio_file_save = './enhanced'
ckpt_path = './checkpoints/best.model'

test_name = os.listdir(noisy_file_path)
test_name.sort(key=lambda x: int(x[7:-4]))


test_data = TestDataset( clean_file_path, noisy_file_path, frame_size=512, frame_shift=256)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=TestCollate())


model = Net()
model = nn.DataParallel(model, device_ids=[0])
checkpoint = Checkpoint()
checkpoint.load(ckpt_path)
model.load_state_dict(checkpoint.state_dict)
model.cuda()
print(checkpoint.start_epoch)
print(checkpoint.best_val_loss)
print(numParams(model))


# test function
def evaluate(net, eval_loader):
    net.eval()

    print('********Starting metrics evaluation on test dataset**********')
    total_stoi = 0.0
    total_ssnr = 0.0
    total_pesq = 0.0
    total_csig = 0.0
    total_cbak = 0.0
    total_covl = 0.0

    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        for k, (features, labels) in enumerate(test_loader):
            features = features.cuda()  # [1, 1, num_frames,frame_size]
            labels = labels.cuda()  # [signal_len, ]

            output = net(features)  # [1, 1, sig_len_recover]
            output = output.squeeze()  # [sig_len_recover, ]

            # keep length same (output label)
            output = output[:labels.shape[-1]]

            eval_loss = torch.mean((output - labels) ** 2)
            total_eval_loss += eval_loss.data.item()

            est_sp = output.cpu().numpy()
            cln_raw = labels.cpu().numpy()

            eval_metric = eval_composite(cln_raw, est_sp, sr)

            # st = get_stoi(cln_raw, est_sp, sr)
            # pe = get_pesq(cln_raw, est_sp, sr)
            # sn = snr(cln_raw, est_sp)
            total_pesq += eval_metric['pesq']
            total_ssnr += eval_metric['ssnr']
            total_stoi += eval_metric['stoi']
            total_cbak += eval_metric['cbak']
            total_csig += eval_metric['csig']
            total_covl += eval_metric['covl']

            wavfile.write(os.path.join(audio_file_save, 'enhanced_' + test_name[k].split('_')[-1]), sr, est_sp.astype(np.float32))

            count += 1
        avg_eval_loss = total_eval_loss / count

    return avg_eval_loss, total_stoi / count, total_pesq / count, total_ssnr / count, total_csig / count, total_cbak / count, total_covl / count


def eva_noisy(file_path):
    print('********Starting metrics evaluation on raw noisy data**********')
    total_stoi = 0.0
    total_ssnr = 0.0
    total_pesq = 0.0
    total_csig = 0.0
    total_cbak = 0.0
    total_covl = 0.0
    count = 0

    with open(file_path, 'r') as eva_file_list:
        file_list = [line.strip() for line in eva_file_list.readlines()]

    for i in range(len(file_list)):
        filename = file_list[i]
        reader = h5py.File(filename, 'r')

        noisy_raw = reader['noisy_raw'][:]
        cln_raw = reader['clean_raw'][:]

        eval_metric = eval_composite(cln_raw, noisy_raw, sr)

        total_pesq += eval_metric['pesq']
        total_ssnr += eval_metric['ssnr']
        total_stoi += eval_metric['stoi']
        total_cbak += eval_metric['cbak']
        total_csig += eval_metric['csig']
        total_covl += eval_metric['covl']

        count += 1

    return total_stoi / count, total_pesq / count, total_ssnr / count, total_cbak / count, total_csig / count, total_covl / count


avg_eval, avg_stoi, avg_pesq, avg_ssnr, avg_csig, avg_cbak, avg_covl = evaluate(model, test_loader)

# avg_stoi, avg_pesq, avg_ssnr, avg_cbak, avg_csig, avg_covl = eva_noisy(test_file_list_path)

# print('Avg_loss: {:.4f}'.format(avg_eval))
print('STOI: {:.4f}'.format(avg_stoi))
print('SSNR: {:.4f}'.format(avg_ssnr))
print('PESQ: {:.4f}'.format(avg_pesq))
print('CSIG: {:.4f}'.format(avg_csig))
print('CBAK: {:.4f}'.format(avg_cbak))
print('COVL: {:.4f}'.format(avg_covl))