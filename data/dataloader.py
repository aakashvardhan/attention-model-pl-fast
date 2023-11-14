from lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.dataset import *
from torch.utils.data import DataLoader
from pathlib import Path
from utils.data_helper import *
import os


class DataModule(LightningDataModule):
    def __init__(self, config,
                 num_workers: int = 4):
        super().__init__()
        self.config = config
        self.num_workers = num_workers
        self.batch_size = config['batch_size']
    def prepare_data(self):
        # Download the data
        # This method is only called once per GPU
        # We will download the data in the data_helper.py file
        load_data(self.config)

    def setup(self, stage=None):
        # Load the data
        # This method is called on every GPU
        # We will load the data in the data_helper.py file
        self.ds_raw = load_data(self.config)
        
        self.tokenizer_src = get_or_build_tokenizer(self.config,self.ds_raw, 
                                                    self.config['lang_src'])
        self.tokenizer_tgt = get_or_build_tokenizer(self.config,self.ds_raw, 
                                                    self.config['lang_tgt'])
        
        self.ds_raw = clean_long_fr_text(self.config,self.ds_raw)
        # Split the data into train, val, and test
        # We will split the data in the data_helper.py file
        self.train_ds_raw, self.val_ds_raw = split_data(self.ds_raw)
        
        self.train_ds = BillingualDataset(self.train_ds_raw,
                                            self.tokenizer_src,
                                            self.tokenizer_tgt,
                                            self.config['lang_src'],
                                            self.config['lang_tgt'],
                                            self.config['seq_len'])
        self.val_ds = BillingualDataset(self.val_ds_raw,
                                            self.tokenizer_src,
                                            self.tokenizer_tgt,
                                            self.config['lang_src'],
                                            self.config['lang_tgt'],
                                            self.config['seq_len'])
        
        
        # find the max length of each sentence in the source and target language
        max_len_src = 0
        max_len_tgt = 0

        for item in self.ds_raw:
            src_ids = self.tokenizer_src.encode(item['translation'][self.config['lang_src']]).ids
            tgt_ids = self.tokenizer_tgt.encode(item['translation'][self.config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length of source language: {max_len_src}")
        print(f"Max length of target language: {max_len_tgt}")
        
    def collate_fn(self, batch):
        return dynamic_collate_fn(batch, self.tokenizer_tgt)

    def train_dataloader(self):
        # Create the train dataloader
        # This method is called on every GPU
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        # Create the validation dataloader
        # This method is called on every GPU
        return DataLoader(self.val_ds,
                          batch_size=1,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers)
        
        