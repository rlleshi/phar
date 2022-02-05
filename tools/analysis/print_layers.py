import argparse

from mmaction.apis import init_recognizer


def parse_args():
    parser = argparse.ArgumentParser(prog='model layer printer')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def print_layers(model, layer_name):
    if len(model._modules) == 0:
        print(layer_name)
    else:
        for key in model._modules:
            name = key if len(layer_name) == 0 else layer_name + '/' + key
            print_layers(model._modules[key], name)


def main():
    args = parse_args()
    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    print_layers(model, '')


if __name__ == '__main__':
    main()
