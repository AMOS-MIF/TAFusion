import glob
from os.path import join
import time
from glob import glob
from os import listdir
import torch
import numpy as np
from PIL import Image
from Fusion_Framework.utils.util import setup_seed
from Compute_Metrics.temp import fun
from Net.models.model import Encoder_Decoder
from Fusion_Framework.utils.util import normalize_01
from FusionNet.config.fusion_config_test import args
from Fusion_Framework.FusionNet.FusionLayer import Fusion_network
from Fusion_Framework.Compute_Metrics.img_read_save import img_save
from FusionNet.utils import *
# -------------------------设置随机数种子--------------------------
setup_seed(42)
# -----------------------设置环境------------------------
device = 'cuda:0'
# -----------------------加载数据集------------------------------------

Path_MRI=''
Path_CT=''
Path_MRI1=''
Path_PET=''
Path_MRI2=''
Path_SPECT='


# ---------------------------定义模型-------------------------------------
batch_size = args.batch_size
nb_filter = [16,32,64, 128]

with torch.no_grad():
    model = Encoder_Decoder()
    model_path = args.auto_encoder
    model.load_state_dict(torch.load(model_path))
    model.eval()

# ----------------------定义融合网络-----------------------
with torch.no_grad():
    fusion_model = Fusion_network(nb_filter)
    if args.fusion_model != None :
        model_path = args.fusion_model
        fusion_model.load_state_dict(torch.load(model_path))
    fusion_model.eval()

# -----------------cuda-----------------------
model = model.to(device)
fusion_model = fusion_model.to(device)
if args.resume_fusion_model is not None:
    print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_fusion_model))
    fusion_model.load_state_dict(torch.load(args.resume_fusion_model))

def test():
    print('Start testing.....')

    resume_fusion_model = "./checkpoints/0.4/model.pth"
    print(resume_fusion_model)
    fusion_model.load_state_dict(torch.load(resume_fusion_model))

    test_one_epoch(Path_MRI,Path_CT,0)
    test_one_epoch(Path_MRI1,Path_PET,1)
    test_one_epoch(Path_MRI2,Path_SPECT,2)
    fun(0)
    fun(1)
    fun(2)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp",".gif"])


def test_one_epoch(Path_A,Path_B,id):
    images_list1 = glob(Path_A + '*.png')
    images_list0 = glob(Path_B + '*.png')
    name1 = []
    name0 = []
    images_list1.sort()
    images_list0.sort()

    index = 0
    for i, image_path in enumerate(images_list1):
        name1.append(image_path)
    for i, image_path in enumerate(images_list0):
        name0.append(image_path)

    MRI_filenames = [join(Path_A, x) for x in listdir(Path_A) if is_image_file(x)]
    CT_filenames = [join(Path_B, x) for x in listdir(Path_B) if is_image_file(x)]
    MRI_name = [x for x in listdir(Path_A) if is_image_file(x)]

    trans=mri_transform()
        
    for i in range(len(MRI_filenames)):
        img0 = Image.open(CT_filenames[i]).convert('YCbCr')
        y1 = Image.open(MRI_filenames[i]).convert('L')
        y0, cb0, cr0 = img0.split()

        mri=trans(y1).to(device)
        ct=trans(y0).to(device)

        mri=torch.unsqueeze(mri,dim=1)
        ct=torch.unsqueeze(ct,dim=1)

        en_ct = model.encoder(ct)
        en_mri = model.encoder(mri)

        outputs, _ = fusion_model(en_ct, en_mri, id)

        output = model.decoder(outputs)
        output = normalize_01(output)


        with torch.no_grad():

            result_path='./output/result/'
            if id==0:
                result_path=result_path+'MRI-CT/'
            elif id==1:
                result_path=result_path+'MRI-PET/'
            else:
                result_path=result_path+'MRI-SPECT/'

            output = np.squeeze((output * 255).cpu().numpy())
            img_save(output, MRI_name[i].split(sep='.')[0], result_path)


            index += 1

if __name__ == '__main__':
    test()

