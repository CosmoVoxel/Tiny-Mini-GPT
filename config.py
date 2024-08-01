from dataclasses import asdict, dataclass
import json
from typing import Dict
from tiktoken import get_encoding


@dataclass
class DecoderConfig:
    layers_d : int = 4
    heads : int = 4
    emb_d : int =  128
    ff_d : int =  emb_d*4
    max_seq_len : int = 1024
    # Encodings [cl100k_base,p50k_base,r50k_base]
    encoding_name: str = 'r50k_base'
    tokenizer =  get_encoding(encoding_name=encoding_name)
    vocab_size : int = 50304  #tokenizer.n_vocab

 
    att_drop: float = 0.1
    ff_drop: float = 0.15

    att_out_drop: float = 0
    ff_out_drop: float = 0

    out_drop: float = 0

    dataset_name:str = "war_and_peace"
    num_workers: int = 8
    epochs: int = 200
    batch_size: int = 6
    grad_accum: int = 1
    lr: float = 6e-4
    valid_freq: int = 1
    bias = False


    def print_all_params(self):
        for i in self.__dict__:
            print(i,":",self.__dict__[i])

    def change_encoding(self):
        self.tokenizer = get_encoding(encoding_name=self.encoding_name)
        self.vocab_size = self.tokenizer.n_vocab
        return self
    
    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict):
        config = cls()
        config.__dict__.update(data)
        return config



def save_config(config: DecoderConfig, file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)

def load_config(file_path: str) -> DecoderConfig:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return DecoderConfig.from_dict(data)



