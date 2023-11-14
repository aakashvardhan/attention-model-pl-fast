from typing import Any
import torch
# import sys
# sys.path.append('../attention-model-pl-fast/models/')
import torch.nn as nn
import torchmetrics
import lightning as pl
from models.attention_model import *
from config import *
import random
from utils.model_helper import *


class AttentionModel(pl.LightningModule):
    def __init__(
        self,
        config,
        tokenizer_src,
        tokenizer_tgt,
        train_dataloader,
        one_cycle_best_LR: float = 1e-4,
        learning_rate: float = 1e-4,
        num_examples: int = 3,
        eps: float = 1e-9,
    ):
        super().__init__()
        
        self.config = config
        self.one_cycle_best_LR = one_cycle_best_LR
        self.learning_rate = learning_rate
        self.num_examples = num_examples
        self.eps = eps
        self.training_step_outputs = []
        
        self.save_hyperparameters()
        
        self.tk_src = tokenizer_src
        self.tk_tgt = tokenizer_tgt
        self.train_dataloader = train_dataloader
        
        self.src_vocab_size = self.tk_src.get_vocab_size()
        self.tgt_vocab_size = self.tk_tgt.get_vocab_size()
        
        self.model : Transformer = build_transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            config['seq_len'],
            config['seq_len'],
            config['d_model'])
        
    def forward(self, batch):
        encoder_input = batch['encoder_input'] # (batch_size, seq_len)
        decoder_input = batch['decoder_input'] # (batch_size, seq_len)

        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']
        
        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
        proj_output = self.model.project(decoder_output) # (batch_size, seq_len, vocab_size)
        return proj_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate,
                                     eps=self.eps)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.one_cycle_best_LR,
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.trainer.max_epochs,
            pct_start=0.2,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear")
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def loss_fn(self, proj_output, label):
        # Define the loss function
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tk_tgt.token_to_id("[PAD]"), label_smoothing=0.1)
        loss = loss_fn(proj_output.view(-1, self.tgt_vocab_size), label.view(-1))
        return loss
    
    def training_step(self, batch, batch_idx):
        proj_output = self(batch)
        label = batch['label'] # (batch_size, seq_len)
        loss = self.loss_fn(proj_output, label)
        self.log_dict({"train_loss":loss}, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.training_step_outputs.append(loss)
        
        return loss
    
    def on_train_epoch_start(self):
        # Clear or reset the training_step_outputs at the beginning of each epoch
        self.training_step_outputs.clear()
    
    def on_train_epoch_end(self):
        if self.training_step_outputs:
            epoch_mean_loss = torch.stack(self.training_step_outputs).mean()
            self.log("train_loss", epoch_mean_loss, on_epoch=True)
        else:
            print("Warning: No outputs to stack for epoch mean loss calculation.")
            
    def on_load_checkpoint(self, checkpoint):
        # Clear or restore self.training_step_outputs as needed
        self.training_step_outputs.clear() 

        
    def validation_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        encoder_mask = batch['encoder_mask']
        
        # check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(self.model, 
                                  encoder_input, 
                                  encoder_mask, 
                                  self.tk_src, 
                                  self.tk_tgt, 
                                  self.config['seq_len'])
        
        source_text = batch['src_text'][0]
        target_text = batch['tgt_text'][0]
        
        model_out_text = self.tk_tgt.decode(model_out.detach().cpu().numpy())

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(model_out_text)
        
        #log the validation loss
        
        
    def on_validation_epoch_start(self):
        self.source_texts = []
        self.expected = []
        self.predicted = []
        
    def on_validation_epoch_end(self):
        
        # Print 5 examples
        for _ in range(self.num_examples):
            idx = random.randint(0, len(self.source_texts) - 1)
            print("-" * 80)
            print(f"{f'SOURCE: ':>12}{self.source_texts[idx]}")
            print(f"{f'TARGET: ':>12}{self.expected[idx]}")
            print(f"{f'PREDICTED: ':>12}{self.predicted[idx]}")
    
        
        cer_metric = torchmetrics.text.CharErrorRate()
        wer_metric = torchmetrics.text.WordErrorRate()
        bleu_metric = torchmetrics.text.BLEUScore()
        
        # Character Error Rate
        cer = cer_metric(self.predicted, self.expected)

        # Word Error Rate
        wer = wer_metric(self.predicted, self.expected)

        # BLEU Score   
        bleu = bleu_metric(self.predicted, self.expected)
        
        # Log the validation loss dictionary
        self.log_dict({"val_cer": cer, "val_wer": wer, "val_bleu": bleu}, 
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        
    