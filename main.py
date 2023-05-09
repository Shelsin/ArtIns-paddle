import argparse
import os
import paddle
import paddle.nn as nn
from PIL import Image
from paddle.vision.transforms import transforms
from tqdm import tqdm
import numpy as np

import copy
from utils import to_tensor
from utils import postprocess
from utils import HtmlPageVisualizer

import src.AdaIN_utils as AdaIN_utils
import src.SANet_utils as SANET_utils

def parse_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='datasets/content/1.jpg', help='File path to the content image')
    parser.add_argument('--style_dir', type=str, default='datasets/style/1.jpg', help='File path to the style image')
    parser.add_argument('--model_name', type=str, default='AdaIN', help='you can choose one from AdaIN/SANet')
    parser.add_argument('--ext', default='.jpg', help='The extension name of the image')

    # save images or not
    parser.add_argument('--save_flag', action='store_true', default=True,
                        help='whether or not to save the results as img')
    parser.add_argument('--save_dir', type=str, default='output_results',
                        help='Directory to save the visualization pages.')

    # distance
    parser.add_argument('--start_distance', type=float, default=-100.0,
                        help='Start point for manipulation on each semantic.')
    parser.add_argument('--end_distance', type=float, default=100.0,
                        help='Ending point for manipulation on each semantic. ')
    parser.add_argument('--step', type=int, default=41, help='Manipulation step on each semantic. ')
    parser.add_argument('--num_semantics', type=int, default=2, help='Number of semantic boundaries')
    parser.add_argument('--component_dir', type=str, default='component_results', help='Directory to save the component directions.')

    # HTML visualize saving or not
    parser.add_argument('--viz_flag', action='store_true', default=True,
                        help='whether or not show the results  on the HTML page.')
    parser.add_argument('--viz_size', type=int, default=256, help='Size of images to visualize on the HTML page.')
    parser.add_argument('--HTML_results', type=str, default='html_results',
                        help='Directory to save the HTML visualization pages.')

    return parser.parse_args()

def main():
    args = parse_args()
    device = 'gpu' if paddle.get_device().startswith('gpu') else 'cpu'

    # bulid filedir
    if args.save_flag:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.viz_flag:
        if not os.path.exists(args.HTML_results):
            os.makedirs(args.HTML_results)

    # getting image dir
    content_info = args.content_dir
    style_info = args.style_dir

    # initialization
    if args.model_name == 'AdaIN':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([transforms.ToTensor(), normalize])
        model = AdaIN_utils.AdaIN_Model()
        model.replace_parameters(paddle.load("model/AdaIN/model_state.pth"))
        model = model.to(device)
        c = Image.open(content_info)
        s = Image.open(style_info)
        c_tensor = trans(c).unsqueeze(0).to(device)
        s_tensor = trans(s).unsqueeze(0).to(device)

    elif args.model_name == 'SANet':
        vgg, decoder = SANET_utils.SANETmoedl()
        transform = SANET_utils.SANET_Transform(in_planes=512).to(device)
        decoder.eval()
        transform.eval()
        vgg.eval()
        decoder.set_state_dict(paddle.load("model/SANet/decoder_iter_500000.pth"))
        transform.set_state_dict(paddle.load("model/SANet/transformer_iter_500000.pth"))
        vgg.set_state_dict(paddle.load("model/SANet/vgg_normalised.pth"))
        enc_1 = nn.Sequential(*list(vgg.children())[:4]).to(device)  # input -> relu1_1
        enc_2 = nn.Sequential(*list(vgg.children())[4:11]).to(device)  # relu1_1 -> relu2_1
        enc_3 = nn.Sequential(*list(vgg.children())[11:18]).to(device)  # relu2_1 -> relu3_1
        enc_4 = nn.Sequential(*list(vgg.children())[18:31]).to(device)  # relu3_1 -> relu4_1
        enc_5 = nn.Sequential(*list(vgg.children())[31:44]).to(device)  # relu4_1 -> relu5_1
        content_tf = SANET_utils.SA_test_transform()
        style_tf = SANET_utils.SA_test_transform()
        content = content_tf(Image.open(content_info))
        style = style_tf(Image.open(style_info))
        style = paddle.to_tensor(style.unsqueeze(0)).to(device)
        content = paddle.to_tensor(content.unsqueeze(0)).to(device)
        decoder = decoder.to(device)


    with paddle.no_grad():
        if args.model_name == 'AdaIN':
            cF, sF = model.getting_code(c_tensor, s_tensor)

        elif args.model_name == 'SANet':
            Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
            Content5_1 = enc_5(Content4_1)
            Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
            sF = copy.deepcopy(Style4_1)

        distances = np.linspace(args.start_distance, args.end_distance, args.step)
        num_sem = args.num_semantics

        if args.viz_flag:
            vizer = HtmlPageVisualizer(num_rows=num_sem,
                                       num_cols=args.step + 1,
                                       viz_size=args.viz_size)
            headers = [''] + [f'Distance {d:.2f}' for d in distances]
            vizer.set_headers(headers)
            for sem_id in range(num_sem):
                vizer.set_cell(sem_id, 0, text=f'Semantic {sem_id+1:03d}', highlight=True)


        component_name = args.model_name + ".npz"
        component_path = os.path.join(args.component_dir,component_name)
        boundaries = paddle.to_tensor(np.load(component_path)["S"]).to(device)


        for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
            boundary = boundaries[sem_id:sem_id + 1]
            for col_id, d in enumerate(distances, start=1):
                sf_test = copy.deepcopy(sF)
                x = paddle.ones(sf_test.shape).to(device)
                for i in range(sf_test.shape[1]):
                    x[:, i, :, :] = x[:, i, :, :] * boundary[0][i]

                sf_test = sf_test + x * d
                # -------------Test------------#
                if args.model_name == 'AdaIN':
                    code = model.getting_total_code(cF, sf_test)
                    content = model.generate(code)
                    content = AdaIN_utils.denorm(content, device)
                elif args.model_name == 'SANet':
                    Style5_1 = enc_5(sf_test)
                    code = transform(Content4_1, sf_test, Content5_1, Style5_1)
                    content = decoder(code)

                content.clamp(0, 255)
                if args.save_flag:
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    img_name = f'{sem_id + 1:03d}_{col_id:04d}{args.ext}'
                    img_path = os.path.join(args.save_dir, img_name)
                    content_np = postprocess(content.squeeze().numpy())
                    Image.fromarray(content_np).save(img_path)

                # HTML visualization
                if args.viz_flag:
                    content_np = postprocess(content.squeeze().numpy())
                    vizer.set_cell(sem_id, col_id, tensor=to_tensor(content_np))

            if args.viz_flag:
                page_name = f'{args.model_name}_viz_{os.path.basename(content_info).split(".")[0]}_{os.path.basename(style_info).split(".")[0]}.html'
                page_path = os.path.join(args.HTML_results, page_name)
                vizer.save(page_path)


if __name__ == '__main__':
    main()