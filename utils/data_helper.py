import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import random_split

def clean_long_fr_text(config,text):
    src_config = config["lang_src"]
    tgt_config = config["lang_tgt"]
    text = sorted(
        text, key=lambda x: len(x["translation"][src_config])
    )
    text = [item for item in text if len(item["translation"][src_config]) <= 150
            and len(item["translation"][tgt_config]) <= 150]

    text = [item for item in text if len(item["translation"][src_config]) + 10
            >= len(item["translation"][tgt_config])]
    return text


def load_data(config):
    return load_dataset(config['dataset'], f"{config['lang_src']}-{config['lang_tgt']}", split="train")

def split_data(ds_raw,train_size=0.9):
    # Split the data into train, val, and test
    # Keep 90% of the data for training, 10% for validation
    train_ds_size = int(train_size * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    return train_ds_raw, val_ds_raw


def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_all_sentences(ds, lang):
        for item in ds:
            yield item['translation'][lang]

def get_or_build_tokenizer(config,ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def dynamic_collate_fn(batch, tokenizer_tgt):
    # Dynamic batch padding
    # Find max seq_len in batch
    # max_len = max(list(map(lambda x: x["max_len"], batch)))
    enc_len = max([len(item['encoder_input']) for item in batch])
    dec_len = max([len(item['decoder_input']) for item in batch])

    encoder_input = []
    decoder_input = []
    label = []

    for item in batch:
        enc_item = item['encoder_input']
        dec_item = item['decoder_input']
        label_item = item['label']

        # Pad the encoder input
        enc_item = torch.cat(
            [
                enc_item,
                torch.tensor([tokenizer_tgt.token_to_id("[PAD]")] * 
                             (enc_len - len(enc_item)), dtype=torch.int64),
            ],
            dim=0
        )

        # Pad the decoder input
        dec_item = torch.cat(
            [
                dec_item,
                torch.tensor([tokenizer_tgt.token_to_id("[PAD]")] * 
                             (dec_len - len(dec_item)), dtype=torch.int64),
            ],
            dim=0
        )

        # Pad the label
        label_item = torch.cat(
            [
                label_item,
                torch.tensor([tokenizer_tgt.token_to_id("[PAD]")] * 
                             (dec_len - len(label_item)), dtype=torch.int64),
            ],
            dim=0
        )

        encoder_input.append(enc_item)
        decoder_input.append(dec_item)
        label.append(label_item)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    encoder_mask = (encoder_input != tokenizer_tgt.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1).int()
    # Assume batch_size is the size of the batch, and dec_len is the length of the decoder sequences
    # Assume batch_size is the size of the batch, and dec_len is the length of the decoder sequences
    batch_size = decoder_input.size(0)

    # Generate the causal mask with the correct size
    causal_mask = casual_mask(dec_len).unsqueeze(1).repeat(batch_size, 1, 1,1)

    # Debugging: Print the shapes of the tensors
    # print("causal_mask new shape (after unsqueeze and repeat):", causal_mask.shape)
    decoder_mask = (decoder_input != tokenizer_tgt.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(2).int()
    # print("decoder_mask shape (after unsqueeze):", decoder_mask.shape)
    # print("causal_mask shape (after repeat):", causal_mask.shape)

    # The bitwise AND operation
    decoder_mask = decoder_mask & causal_mask.int()

    
    label = torch.stack(label)
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]

    return {
        "encoder_input": encoder_input, # (batch_size, seq_len)
        "decoder_input": decoder_input, # (batch_size, seq_len)
        "encoder_mask": encoder_mask, # (batch_size, 1, 1, seq_len)
        "decoder_mask": decoder_mask, # (batch_size, 1, seq_len, seq_len)
        "label": label, # (batch_size, seq_len)
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }


