import argparse
import torch
import time

from pathlib import Path

from models.definitions import Vgg19
from utils.losses import get_loss_func
from utils.images import load_img, save_img
from utils import time_execution


def neural_style_transfer(target_img, style_img, content_img, config):
    device = config['device']
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    model = Vgg19()
    model.to(device)
    model.eval()  # model is not trained, the target image is

    target_img = target_img.to(device).requires_grad_()
    style_img = style_img.to(device)
    content_img = content_img.to(device)

    content_feature_maps = model(content_img)
    style_feature_maps = model(style_img)

    loss = get_loss_func(content_feature_maps, style_feature_maps,
                         model.content_feature_maps_idx, model.style_feature_maps_indices,
                         config['alpha'], config['beta'])
    # the paper used stochastic gradient descent
    optimizer = torch.optim.Adam((target_img,), config['lr'])

    for i in range(config['epochs_num']):
        optimizer.zero_grad()

        target_feature_maps = model(target_img)

        total_loss, content_loss, style_loss = loss(target_feature_maps)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if i % 25 == 0:
                print(
                    f"epoch {i}    total_loss: {total_loss.item()}   content_loss: {content_loss.item()}    style_loss: {style_loss.item()}")
            if config['save_freq'] is not None and i % config['save_freq'] == 0:
                save_img(
                    target_img, f"{config['save_dir']}/{config['base_filename']}_{i}.jpg")

    with torch.no_grad():
        save_img(
            target_img, f"{config['save_dir']}/{config['base_filename']}_final.jpg")


@time_execution
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--style-img-path', required=True, type=str)
    parser.add_argument('--content-img-path', required=True, type=str)
    parser.add_argument('--height', type=int,
                        help="Height of the generated image")
    parser.add_argument('--width', type=int,
                        help="Width of the generated image")
    parser.add_argument('--start-img',
                        choices=['random', 'style', 'content'],
                        default='random')
    parser.add_argument('--alpha', default=1e1, type=float)
    parser.add_argument('--beta', default=1e10, type=float)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--epochs-num', default=3000, type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--save-freq', type=int)
    parser.add_argument('--save-dir', type=str)
    # according to my experience the generated image is better when the below option is passed
    parser.add_argument('--resize-style-img', action='store_true')

    args = parser.parse_args()

    style_img_path = args.style_img_path
    content_img_path = args.content_img_path
    height = args.height
    width = args.width
    start_img = args.start_img
    resize_style_img = args.resize_style_img

    default_save_dir = f"nst/{start_img}_h_{height}_w_{width}_lr_{args.lr}_a_{args.alpha}_b_{args.beta}/{int(time.time())}"

    config = {
        'alpha': args.alpha,
        'beta': args.beta,
        'lr': args.lr,
        'epochs_num': args.epochs_num,
        'device': args.device,
        'save_freq': args.save_freq,
        'save_dir': args.save_dir if args.save_dir is not None else default_save_dir,
        'base_filename': f"{Path(content_img_path).stem}_{Path(style_img_path).stem}"
    }

    content_img = load_img(content_img_path, height, width)
    if resize_style_img:
        _, content_img_height, content_img_width = content_img.shape
        style_img = load_img(
            style_img_path, content_img_height, content_img_width)
    else:
        style_img = load_img(style_img_path)

    # the paper used random initialization
    if start_img == 'random':
        target_img = torch.randn_like(content_img)
    elif start_img == 'content':
        target_img = content_img.clone().detach()
    elif start_img == 'style':
        _, content_img_height, content_img_width = content_img.shape
        target_img = load_img(
            style_img_path, content_img_height, content_img_width)
        target_img = target_img.clone().detach()

    neural_style_transfer(target_img, style_img, content_img, config)


if __name__ == '__main__':
    main()
