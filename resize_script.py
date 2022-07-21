import argparse

from utils.images import load_resize_save_img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--height', required=True, type=int)
    parser.add_argument('--width', type=int)

    args = parser.parse_args()

    load_resize_save_img(args.path, args.height, args.width)


if __name__ == '__main__':
    main()
