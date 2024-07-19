from torch.utils.data import DataLoader
from BERT import BERTMini
from pretrain import BERTTrainer
from dataset.dataset import BERTDatase
from dataset.vocab import WordVocab
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--train_dataset', default='books.txt')
    parser.add_argument('-t', '--test_dataset', default=None)
    parser.add_argument('-v', '--vocab_path', default='vocab.pkl')
    parser.add_argument('-o', '--output_path', default='output/bert.model')

    parser.add_argument('-l', '--layers', type=int, default=6)
    parser.add_argument('-hs', '--hidden', type=int, default=256)
    parser.add_argument('-a', '--attn_heads', type=int, default=4)
    parser.add_argument('-s', '--seq_len', type=int, default=256)

    parser.add_argument('-b', '--batch_size', type=int, default=24)
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-w', '--num_workers', type=int, default=4)

    args = parser.parse_args()

    print("Loading Vocab")
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset")
    train_dataset = BERTDatase(
        corpus_path=args.train_dataset,
        vocab=vocab,
        seq_len=args.seq_len
    )

    print("Creating Dataloader")
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print("Building BERT model")
    bert = BERTMini(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads
    )

    print("Creating BERT Trainer")
    trainer = BERTTrainer(
        bert=bert,
        vocab_size=len(vocab),
        train_dataloader=train_data_loader,
    )

    print("Training Start")
    for epoch in range(args.epoch):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)
