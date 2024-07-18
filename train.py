from torch.utils.data import DataLoader
from BERT import BERTMini
from pretrain import BERTTrainer
from dataset.dataset import BERTDatase
from dataset.vocab import WordVocab
import argparse

if __name__ == "__main__":

    print("Loading Vocab")
    vocab = WordVocab.load_vocab('./vocab.pkl')
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset")
    train_dataset = BERTDatase(
        corpus_path='./books.txt',
        vocab=vocab,
        seq_len=512,
    )

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=6)

    print("Building BERT model")
    bert = BERTMini(
        vocab_size=len(vocab),
    )

    print("Creating BERT Trainer")
    trainer = BERTTrainer(
        bert=bert,
        vocab_size=len(vocab),
        train_dataloader=train_data_loader,
    )

    print("Training Start")
    for epoch in range(1):
        trainer.train(epoch)
        trainer.save(epoch)
