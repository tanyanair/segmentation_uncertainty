from bunet.data.tfr_data_provider import BrainVolumeDataProvider as DataProvider
from bunet.models.bunet import BUnet
from bunet.training.train_tfr import Trainer
from argparse import ArgumentParser
from shutil import copy
import json
from os import makedirs


def main(args):
    out_dir = args.output
    with open(args.config, 'r') as f:
        cfg = json.loads(f.read())
    makedirs(out_dir, exist_ok=True)
    copy(args.config, out_dir)
    expt_cfg = cfg['expt']
    model_cfg = cfg['model']

    net = BUnet(nb_ch=model_cfg['nb_ch'],
                nb_kers=model_cfg['nb_kers'],
                nb_mc=model_cfg['nb_mc'],
                depth=model_cfg['depth'],
                weight_decay=model_cfg['wd'],
                batch_size=expt_cfg['batch_size'])

    train_ds = DataProvider(expt_cfg['data_path'],
                            {'mode': 'train', 'shuffle': True if expt_cfg['shuffle'] is 1 else False})
    valid_ds = DataProvider(expt_cfg['data_path'], {'mode': 'valid', 'shuffle': False})
    train_gen = train_ds.get_generator(expt_cfg['batch_size'], expt_cfg['nb_epochs'])
    valid_gen = valid_ds.get_generator(expt_cfg['batch_size'], expt_cfg['nb_epochs'])

    trainer = Trainer(net,
                      opt_kwargs={'lr': model_cfg['lr'], 'decay': model_cfg['lr_decay']},
                      batch_size=expt_cfg['batch_size'])

    trainer.train(train_gen,
                         valid_gen,
                         nb_val_steps=expt_cfg['nb_val_steps'],
                         output_path=out_dir,
                         steps_per_epoch=expt_cfg['steps_per_epoch'],
                         epochs=expt_cfg['nb_epochs'],
                         dropout=1 - model_cfg['dr'],
                         restore_path=expt_cfg['restore_path'],
                         viz=True if expt_cfg['viz'] is 1 else False,
                         cca_thresh=0.5,
                         class_weight=expt_cfg['cw'])


def _parser():
    usage = ''
    parser = ArgumentParser(prog='train_unet', usage=usage)
    parser.add_argument('-o', '--output', help='Output Directory', required=True)
    parser.add_argument('-c', '--config', help='json Configuration File', required=True)
    return parser


if __name__ == '__main__':
    main(_parser().parse_args())
