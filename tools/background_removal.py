import glob
import os
import os.path as osp
import random
import shutil
import string
from argparse import ArgumentParser
from pathlib import Path

import moviepy.editor as mpy
import torch
from cv2 import cv2
from data.background.data_loader import RescaleT, SalObjDataset, ToTensorLab
from PIL import Image
from rich.console import Console
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from u2net import U2NET

CONSOLE = Console()
TEMP_DIR = 'temp_dir'
EXT = ['.mp4']


def gen_id(size=8):
    """Generate a random id."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


def generate_structure(out):
    for c in [
            'walk', 'run', 'jump', 'stamp', 'contract_expand', 'tiptoe',
            'swing_upper_body', 'rotate', 'fall'
    ]:
        Path(osp.join(out, c)).mkdir(parents=True, exist_ok=True)


def min_max_norm(x):
    """Min-Max normalization using PyTorch."""
    maX = torch.max(x)
    mIn = torch.min(x)
    return (x - mIn) / (maX - mIn)


def save_output(image_name, pred, out_dir):
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()
    im = Image.fromarray(pred * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    aaa = img_name.split('.')
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + '.' + bbb[i]
    imo.save(osp.join(out_dir, imidx) + '.png')


def save_video(frames, fps, final_dir, out_vid):
    frames = glob.glob(osp.join(final_dir, '*'))
    frames = sorted(frames, key=lambda x: int(x.split('/')[-1][:-4]))
    out = mpy.ImageSequenceClip(frames, fps=fps)
    out.write_videofile(out_vid, remove_temp=True)


def parse_args():
    parser = ArgumentParser(prog='video background removal using u2net'
                            'salient object detection')
    parser.add_argument('src_dir', help='src dir')
    parser.add_argument('out_dir', help='out dir')
    parser.add_argument(
        '--model',
        type=str,
        default='../models/salient-object-det/u2net.pth',
        help='path to background extraction model')
    parser.add_argument(
        '--level',
        type=int,
        default=3,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='device to use, `None` for no gpus')
    args = parser.parse_args()
    return args


def inference(model, img_list, sal_obj_dataloader, out_dir):
    for i, data in enumerate(sal_obj_dataloader):
        inputs = data['image']
        inputs = inputs.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        d1, d2, d3, d4, d5, d6, d7 = model(inputs)

        # normalize
        pred = d1[:, 0, :, :]
        pred = min_max_norm(pred)

        # save results intermidetly
        save_output(img_list[i], pred, out_dir)
        del d1, d2, d3, d4, d5, d6, d7


def bitwise_sub(orig_frames, salient_res, out_dir):
    for i in range(len(orig_frames)):
        orig = cv2.imread(orig_frames[i])
        u2net = cv2.imread(salient_res[i])
        result = cv2.subtract(u2net, orig)
        cv2.imwrite(
            osp.join(out_dir, orig_frames[i].split(os.sep)[-1]), result)


def extract_obj(orig_frames, sub_frames, out_dir):
    for i in range(len(orig_frames)):
        orig = (Image.open(orig_frames[i])).convert('RGBA')
        sub = (Image.open(sub_frames[i])).convert('RGBA')
        orig_data = orig.getdata()
        sub_data = sub.getdata()
        result = []

        for j in range(sub_data.size[0] * sub_data.size[1]):
            # replace black pixels with a transparent pixel
            if sub_data[j][0] == 0 and sub_data[j][1] == 0 and sub_data[j][
                    2] == 0:
                result.append((255, 255, 255, 0))
            else:
                result.append(orig_data[j])

        sub.putdata(result)
        sub.save(osp.join(out_dir, orig_frames[i].split(os.sep)[-1]))


def background_removal(video, model, out_dir):
    CONSOLE.log(video)
    v_name = video.split(os.sep)[-1]
    # label = video.split(os.sep)[-2]
    # out_vid = osp.join(out_dir, label, v_name)
    out_vid = osp.join(out_dir, v_name)
    if osp.exists(out_vid):
        CONSOLE.print(f'{out_vid} already exists', style='yellow')
        return

    frame_dir = osp.join(TEMP_DIR, gen_id(4))
    salient_dir = osp.join(TEMP_DIR, gen_id(4))
    sub_dir = osp.join(TEMP_DIR, gen_id(4))
    final_dir = osp.join(TEMP_DIR, gen_id(4))
    Path(frame_dir).mkdir(parents=True, exist_ok=True)
    Path(salient_dir).mkdir()
    Path(sub_dir).mkdir()
    Path(final_dir).mkdir()

    # extract frames
    video_cv = cv2.VideoCapture(video)
    fps = video_cv.get(cv2.CAP_PROP_FPS)
    c = 0
    while (video_cv.isOpened()):
        flag, frame = video_cv.read()
        if not flag:
            break
        cv2.imwrite(osp.join(frame_dir, str(c)) + '.png', frame)
        c += 1

    frames = glob.glob(osp.join(frame_dir, '*'))
    # load dataset
    sal_obj_dataset = SalObjDataset(
        img_name_list=frames,
        lbl_name_list=[],
        # use crop to focus the person
        transform=transforms.Compose([RescaleT(320),
                                      ToTensorLab(flag=0)]))
    sal_obj_dataloader = DataLoader(
        sal_obj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # salient object detection using u2net
    inference(model, frames, sal_obj_dataloader, salient_dir)
    if len(os.listdir(salient_dir)) == 0:
        CONSOLE.print(
            f'Error with salient object detection for {video}',
            style='bold red')
        return

    # bitwise substraction between u2net result and orig image
    # improves final performance
    bitwise_sub(frames, glob.glob(osp.join(salient_dir, '*')), sub_dir)

    # extract main object and save final result
    extract_obj(frames, glob.glob(osp.join(sub_dir, '*')), final_dir)
    save_video(glob.glob(osp.join(final_dir, '*')), fps, final_dir, out_vid)

    shutil.rmtree(frame_dir)
    shutil.rmtree(salient_dir)
    shutil.rmtree(sub_dir)
    shutil.rmtree(final_dir)


def main():
    args = parse_args()
    if args.device != 'None':
        torch.cuda.set_device(torch.device(args.device))
    # generate_structure(args.out_dir)
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

    model = U2NET(3, 1)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    CONSOLE.log('Loaded model')

    if args.level == 1:
        video_list = glob.glob(osp.join(args.src_dir, '*'))
    elif args.level == 2:
        video_list = glob.glob(osp.join(args.src_dir, '*', '*'))
    else:
        video_list = glob.glob(osp.join(args.src_dir, '*', '*', '*'))
    video_list = [
        video for video in video_list if any(video.endswith(e) for e in EXT)
    ]
    # randomize
    random.shuffle(video_list)

    CONSOLE.log(
        'Removing background through salient object detection for'
        f' {len(video_list)} videos',
        style='bold green')
    for video in tqdm(video_list):
        background_removal(video, model, args.out_dir)


if __name__ == '__main__':
    main()
