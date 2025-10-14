import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils
import numpy as np
import modules
import torch
import torch.nn as nn
import datasets
import yaml
import pdb
import json
import faulthandler
faulthandler.enable()

from seq_scripts import seq_train, seq_eval
import slr_network

class SLRProcessor(object):
    def __init__(self, arg):
        super().__init__()

        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval
        )
        self.dataset = {}
        self.data_loader = {}

        self.load_dataset_info()
        with open(self.arg.dataset_info['dict_path'], 'r') as f:
            self.gloss_dict = json.load(f)
        self.model, self.optimizer = self.loading()
        self.best_dev_wer = 1000
        self.tasks = self.arg.dataset[-2:]
    
    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model = self.build_module(self.arg.model_args)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer
    
    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)['model_state_dict']
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        model.load_state_dict(state_dict, strict=False)
    
    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size
            if mode == "train"
            else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )

    def build_module(self, args):
        model_class = getattr(slr_network, self.arg.model)
        model = model_class(
            **args,
            gloss_dict = self.gloss_dict
        )
        return model

    def load_data(self):
        print("Loading data")
        self.feeder = getattr(datasets, self.arg.feeder)
        # dataset_list = zip(
        #     ["train_dev", "test"], [True, False]
        # )
        dataset_list = zip(
            ["train", "dev", "test"], [True, False, False]
        )
        g2i_dict = {k: v['index'] for k, v in self.gloss_dict['gloss2id'].items()}
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["mode"] = mode
            arg["transform_mode"] = train_flag
            arg["dataset"] = self.arg.dataset
            self.dataset[mode] = self.feeder(gloss_dict=g2i_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(
                self.dataset[mode], mode, train_flag
            )
        print("Loading data finished.")
    
    def load_dataset_info(self):
        with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", 'r') as f:
            self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    
    def judge_save_eval(self, epoch):
        save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0.5 * self.arg.num_epoch)
        # save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0)
        eval_model = (epoch % self.arg.eval_interval == 0) and (epoch >= 0)
        return save_model, eval_model

    def save_model(self, epoch, save_path):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
                'rng_state': self.rng.save_rng_state(),
            },
            save_path,
        )

    def custom_save_model(self, dev_wer, epoch, save_dir):
        dirs = os.listdir(save_dir)
        dirs = list(filter(lambda x: x.endswith('.pt'), dirs))
        assert len(dirs) <= 2
        best_path, cur_path = None, None
        for item in dirs:
            if 'best' in item:
                best_path = os.path.join(save_dir, item)
            if 'cur' in item:
                cur_path = os.path.join(save_dir, item)
        if cur_path is not None:
            os.system(f'rm {cur_path}')
        model_path = "{}cur_dev_{:05.2f}_epoch{}_model.pt".format(
                        save_dir, dev_wer, epoch
                    )
        self.save_model(epoch, model_path)
        if best_path is not None:
            if dev_wer <= self.best_dev_wer:
                os.system(f'rm {best_path}')
                model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(
                        save_dir, dev_wer, epoch
                    )
                self.save_model(epoch, model_path)
                self.best_dev_wer = dev_wer
        else:
            model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(
                        save_dir, dev_wer, epoch
                    )
            self.save_model(epoch, model_path)
            self.best_dev_wer = dev_wer

    def train(self):
        self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        for epoch in range(
            self.arg.optimizer_args['start_epoch'], self.arg.num_epoch
        ):
            save_model, eval_model = self.judge_save_eval(epoch)
            # seq_train(self.data_loader['train_dev'], self.model, self.optimizer, self.device,
            #     epoch, self.recoder, **self.arg.train_args
            # )
            seq_train(self.data_loader['train'], self.model, self.optimizer, self.device,
                epoch, self.recoder, **self.arg.train_args
            )

            if eval_model:
                dev_error = self.test('dev', epoch)
                self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_error))
            
            if save_model:
                self.custom_save_model(dev_error, epoch, self.arg.work_dir)

    def test(self, mode, epoch):
        wer = seq_eval(
                self.arg,
                self.data_loader[mode],
                self.model,
                self.device,
                mode,
                epoch,
                self.arg.work_dir,
                self.recoder,
                self.tasks,
                self.arg.evaluate_tool
            )
        return wer

    def start(self):
        if self.arg.phase == 'train':
            self.train()
        elif self.arg.phase == 'test':
            # if self.arg.load_weights is None and self.arg.load_checkpoints is None:
            #     raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            self.test('dev', 6667)
            self.test('test', 6667)
            self.recoder.print_log('Evaluation Done.\n')

if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert k in key
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()

    main_processor = SLRProcessor(args)
    main_processor.start()