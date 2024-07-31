from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from pytorch_lightning import LightningModule,Trainer,LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils
import torch.utils.data
from model import Decoder,generate_tokens
import numpy as np
from my_dataset import *
import argparse
from config import load_config,DecoderConfig
from datasets import load_from_disk


#ARGPARSET 
parser = argparse.ArgumentParser(description='Train GPT model')
parser.add_argument('-d','--dataset', type=str, help='Dataset to train on [war_and_peace,russian,finewebedu]',default="war_and_peace")
parser.add_argument('-l','--log', type=bool,action=argparse.BooleanOptionalAction, help='Enable Logging',default=False)
parser.add_argument('-s','--saving', type=bool,action=argparse.BooleanOptionalAction, help='Enable checkpoint',default=False)
parser.add_argument('-c','--config', type=str, help='Path to config file',default="None")
parser.add_argument('-ts','--train_size',type=float,help="Select maximum training dataset limit",default=1.0)
parser.add_argument('--compile',type=bool,action=argparse.BooleanOptionalAction, help='Enable compiling',default=True)
parser.add_argument('--validation',type=bool,action=argparse.BooleanOptionalAction, help='Enable validation step',default=True)
parser.add_argument('-t','--train', type=bool,action=argparse.BooleanOptionalAction, help='Enable training',default=True)
parser.add_argument('-i','--inference', type=bool,action=argparse.BooleanOptionalAction, help='Enable inference',default=False)
parser.add_argument('-lc','--load_checkpoint',type=str, help='Path to config file',default="None")

args = parser.parse_args()

if args.config != "None":
    config = load_config(args.config)
elif args.config == "None":
    config = DecoderConfig()

def load_asci_logo_from_file(file):
    with open(file) as f:
        return f.read()

class WarAndPeaceDataModule(LightningDataModule):
    def __init__(self,path = 'war_and_piece_untouched.npy',batch_size: int = config.batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            data = np.load(self.path,allow_pickle=True)
            data :np.ndarray = np.concatenate(data)
            data = torch.tensor(data,dtype=torch.uint16)

            if len(data) % config.max_seq_len != 0:
                data = data[:-(len(data) % config.max_seq_len)]
            data = data.view(-1, config.max_seq_len)

            x = data[:, :-1]  # Input sequences
            y = data[:, 1:]   # Target sequences, shifted by one token to the right

            dataset = torch.utils.data.TensorDataset(x, y)

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers=config.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=config.num_workers)

class RussianBooksDataModule(LightningDataModule):
    def __init__(self,batch_size: int = config.batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            data = np.load("all_russian_books.npy",allow_pickle=True)#[0].astype(np.uint32)
            data = np.concatenate(data)
            tar_size = int(input("Select target number of tokens(-1 = ALL): "))
            if tar_size != -1:
                data = torch.tensor(data[0:tar_size])

            data = torch.from_numpy(data)

            print("Total tokens: ",data.size(0) / 1000000," Mil" )

            if len(data) % config.max_seq_len != 0:
                data = data[:-(len(data) % config.max_seq_len)]
            data = data.view(-1, config.max_seq_len)

            x = data[:, :-1]
            y = data[:, 1:]

            dataset = torch.utils.data.TensorDataset(x, y)

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers=config.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=config.num_workers)

class FineWebEduDataset(torch.utils.data.IterableDataset):
    def __init__(self, data: Dataset):
        self.data = data

    def __iter__(self):
        target_tokens = []
        for i in range(len(self.data)):
            tokenized_text = self.data[i]['tokenized_text']
            
            while len(tokenized_text) > 0:
                remaining_tokens = config.max_seq_len - len(target_tokens)
                
                if len(tokenized_text) <= remaining_tokens:
                    target_tokens += tokenized_text
                    tokenized_text = []
                else:
                    target_tokens += tokenized_text[:remaining_tokens]
                    tokenized_text = tokenized_text[remaining_tokens:]
                    yield torch.tensor(target_tokens[:-1]),torch.tensor(target_tokens[1:])
                    target_tokens = []

        if target_tokens:
            yield torch.tensor(target_tokens[:-1]),torch.tensor(target_tokens[1:])


    def __len__(self):
        return len(self.data)

class FineWebEduDataModule(LightningDataModule):
    def __init__(self,batch_size: int = config.batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            data = load_from_disk("./fine-web-edu-1T-tokenized").train_test_split(test_size=0.1,train_size=0.9)

            train_data, val_data = data['train'], data['test']

            self.train_dataset = FineWebEduDataset(train_data)
            self.val_dataset = FineWebEduDataset(val_data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=config.batch_size,num_workers=11)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=config.batch_size,num_workers=11)

if args.dataset == "war_and_peace":
    config.encoding_name = 'r50k_base'
    config.dataset_name = "war_and_peace"
    config.change_encoding()
    datamodule = WarAndPeaceDataModule("war_and_piece_untouched.npy")

elif args.dataset == "russian":
    config.encoding_name = 'cl100k_base'
    config.dataset_name = "russian"
    config.change_encoding()
    datamodule = RussianBooksDataModule()

elif args.dataset == "finewebedu":
    datamodule = FineWebEduDataModule()
    config.encoding_name = 'r50k_base'
    config.dataset_name = "fine_web_edu"

class GPT(LightningModule):
    def __init__(self, config:DecoderConfig):
        super().__init__()
        self.model = Decoder(config)      
        if args.compile:
            self.model = torch.compile(self.model,fullgraph=True,mode='max-autotune')
      
        torch.set_float32_matmul_precision("medium")
        self.config = config
        self.learning_rate = config.lr
        self.save_hyperparameters(ignore=['model'])
        self.total_validation_steps = 0
        
    def training_step(self, batch, batch_idx):
        tensorboard:SummaryWriter = self.logger.experiment
        x:torch.Tensor
        y:torch.Tensor
        x,y = batch
        x = x.long()
        x = self.model(x)
        y = y.long()
        loss = torch.nn.functional.cross_entropy(x.view(-1,self.config.vocab_size),y.view(-1))
        tensorboard.add_scalars("Loss",{"Train Loss":loss},self.global_step) 
        tensorboard.add_scalars("Learning Rate",{"Learning Rate":self.lr_schedulers().get_last_lr()[0]},self.global_step)
        self.log("train_loss",loss,prog_bar=True,logger=False)
        self.log("lr",self.lr_schedulers().get_last_lr()[0],prog_bar=True,logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        tensorboard:SummaryWriter = self.logger.experiment
        self.total_validation_steps += 1
        x:torch.Tensor
        y:torch.Tensor
        x,y = batch
        x = x.long()
        x = self.model(x)
        y = y.long()
        loss = torch.nn.functional.cross_entropy(x.view(-1,self.config.vocab_size),y.view(-1))
        tensorboard.add_scalars("Loss",{"Validation Loss":loss},self.total_validation_steps+(self.global_step - self.total_validation_steps)) 
        self.log("val_loss",loss,prog_bar=True,logger=False)

    def on_validation_epoch_end(self):
        if args.log:
            if config.dataset_name == "war_and_peace" or 'finewebedu':
                start_text = "I am the only one who"
            else:
                start_text = "В начале"
            text = generate_tokens(self.model,config=self.config,n_tokens=32,start_text=start_text)
            self.logger.experiment.add_text("Generated Text",text,self.current_epoch)

    def configure_optimizers(self):
        print("Configuring optimizers")
        optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate,betas=(0.9,0.95),weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,pct_start=0.1,max_lr=self.learning_rate,steps_per_epoch=int(len(datamodule.train_dataloader())*args.train_size),epochs=config.epochs,div_factor=10)
        return [optimizer], {"scheduler":lr_scheduler,
                             "interval":"step",
                             "frequency":1}

model = GPT(config)

if args.train:
    if args.load_checkpoint:
        data = torch.load(args.load_checkpoint,weights_only=False)
        model = GPT(config)
        GPT.load_state_dict(model,data['state_dict']) 
        print(model)

    callbacks = []
    print(load_asci_logo_from_file("logo.txt"))
    if args.saving:
        checkpoints = ModelCheckpoint(
            filename="gpt_{epoch}_{step}",
            save_on_train_epoch_end=True,
            save_last=True,
            save_top_k=-1
        )
        callbacks.append(checkpoints)

    use_logger = None
    if args.log:
        logger = TensorBoardLogger("tb_logs", name="GPT")
        use_logger = logger
        logger.log_hyperparams(config.__dict__)

    config.print_all_params()
    input("PRESS ENTER TO CONTINUE...")


    trainer = Trainer(max_epochs=config.epochs,precision='bf16-true',callbacks=callbacks,logger=use_logger,num_sanity_val_steps=0
                    ,check_val_every_n_epoch=config.valid_freq,gradient_clip_val=1.0,accumulate_grad_batches=1,
                    limit_val_batches=args.train_size,limit_train_batches=args.train_size)
    

    trainer.fit(model,datamodule=datamodule) 

    if args.saving:
        print(checkpoints.best_model_path)

if args.inference:
    if args.load_checkpoint != 'None':
        model = GPT.load_from_checkpoint(args.load_checkpoint)
    elif args.train:
        model = model
    else:
        print('No checkpoint loaded')
        exit(444)

    model = model.model
    prompt = ""
    while prompt != "exit":
        prompt = input("Message: ")
        if prompt != "exit":
            response = generate_tokens(model,1024,config,prompt)
            print(response)
            print("#"*100)