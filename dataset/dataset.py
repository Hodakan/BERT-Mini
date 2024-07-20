from torch.utils.data import Dataset
from dataset.vocab import WordVocab
import tqdm
import torch
import random


class BERTDatase(Dataset):

    def __init__(self, corpus_path, vocab: WordVocab, seq_len, corpus_lines=None, encoding='utf-8'):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            self.lines = []
            for line in tqdm.tqdm(f, desc="Loading Dataset"):
                line = line.split()
                if len(line) > seq_len:
                    continue
                t1, t2 = line[:len(line)//2], line[len(line)//2:]
                self.lines.append([" ".join(t1), " ".join(t2)])

                if corpus_lines is not None and len(self.lines) >= corpus_lines:
                    break

            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        t1 = [self.vocab.cls_index] + t1_random + [self.vocab.sep_index]
        t2 = t2_random + [self.vocab.sep_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] +
                         [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(
            self.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_label.extend(padding)

        output = {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "is_next": is_next_label
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                else:
                    tokens[i] = self.vocab.stoi.get(
                        token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(
                    token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        return self.lines[random.randrange(self.corpus_lines)][1]
