from tf_unet.analyze.BunetAnalyzer import BunetAnalyzer as Analyzer
from tf_unet.data.data_provider import BrainVolumeDataProvider as DataProvider
from tf_unet.models.bunet import BUnet
from os import makedirs
from os.path import join
from argparse import ArgumentParser


def main(args):
    h5_path = args.data
    for i in ['train_img', 'val_img', 'test_img']:
        d = join(args.checkpoint_path, i)
        makedirs(d, exist_ok=True)
    out_dir = args.checkpoint_path

    train_ds = DataProvider(h5_path, {'mode': 'train', 'shuffle': False, 'modalities': [0, 1, 2, 3]})
    valid_ds = DataProvider(h5_path, {'mode': 'valid', 'shuffle': False, 'modalities': [0, 1, 2, 3]})
    test_ds = DataProvider(h5_path, {'mode': 'test', 'shuffle': False, 'modalities': [0, 1, 2, 3]})

    train_gen = train_ds.get_test_generator(1)
    valid_gen = valid_ds.get_test_generator(1)
    test_gen = test_ds.get_test_generator(1)
    net = BUnet(nb_ch=4,
                nb_kers=32,
                nb_mc=10,
                depth=4,
                weight_decay=0.0001,
                loss_fn='adam',
                batch_size=1)

    sigmoid_thresh = 0.5

    train_analyzer = Analyzer(net, args.checkpoint_path, train_gen, join(out_dir, 'train_img'), nb_mc=10)
    valid_analyzer = Analyzer(net, args.checkpoint_path, valid_gen, join(out_dir, 'valid_img'), nb_mc=10)
    test_analyzer = Analyzer(net, args.checkpoint_path, test_gen, join(out_dir, 'test_img'), nb_mc=10)

    train_analyzer.cca('train_stats_thresh{}.csv'.format(sigmoid_thresh), sigmoid_thresh)
    valid_analyzer.cca('valid_stats_thresh{}.csv'.format(sigmoid_thresh), sigmoid_thresh)
    test_analyzer.cca('test_stats_thresh{}.csv'.format(sigmoid_thresh), sigmoid_thresh)


def _parser():
    usage = 'python run_cca.py -i /cim/data/mslaq.hdf5 -c /path/to/checkpoint/ -o /output/directory'
    parser = ArgumentParser(prog='train_unet', usage=usage)
    parser.add_argument('-i', '--data', help='Training HDF5', required=True)
    parser.add_argument('-c', '--checkpoint_path', help='Model Checkpoint', required=True)
    return parser


if __name__ == '__main__':
    main(_parser().parse_args())
