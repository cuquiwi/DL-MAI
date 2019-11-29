import argparse

parser = argparse.ArgumentParser()
parser.add_argument('source_model', type=str,
        help='Source model. One of: VGG16_ImageNet, VGG16_Places')