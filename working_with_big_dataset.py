from datasets import load_dataset, load_from_disk


dataset = load_from_disk("./fine-web-edu-1T",)

print(dataset)

dataset = dataset.sort('token_count')

print("Start tokenizing!")
input("Press Enter to continue...")


from model import DecoderConfig

config = DecoderConfig()
tokenizer = config.tokenizer


print("Encoding name:",config.encoding_name)
print("Vocab size:",tokenizer.n_vocab)

def tokenize_text(text:str):
    return tokenizer.encode(text)

dataset = dataset.map(tokenize_text,num_proc=11)
