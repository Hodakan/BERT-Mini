import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm

from BERT import BERTMini, BERTLM
from optim_schedule import ScheduleOptim


class BERTTrainer:
    """
    Make the pretrained BERT model with two LM training method.
        1. Masked Language Model
        2. Next Sentence Prediction
    """

    def __init__(
        self, bert: BERTMini, vocab_size: int,
        train_dataloader: DataLoader, test_dataloader: DataLoader = None,
        lr: float = 1e-4, betas=(0.9, 0.999),
        weight_decay: float = 0.01, warmup_steps=10000,
        log_freq: int = 10
    ):
        """
        :param bert: BERT model going to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decat param
        :param log_freq: logging frequency of the batch iteration
        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr,
                          betas=betas, weight_decay=weight_decay)
        self.scheduler = ScheduleOptim(
            self.optim, self.bert.hidden, warmup_steps)

        self.loss_fn = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:",
              sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % ('train' if train else 'test', epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            next_sent_output, mask_lm_output = self.model.forward(
                data['bert_input'], data['segment_label'])

            next_loss = self.loss_fn(next_sent_output, data['is_next'])
            mask_loss = self.loss_fn(
                mask_lm_output.transpose(1, 2), data['bert_label'])

            loss = next_loss + mask_loss

            if train:
                self.scheduler.zero_grad()
                loss.backward()
                self.scheduler.step_and_update_lr()

            correct = next_sent_output.argmax(
                dim=-1).eq(data['is_next']).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data['is_next'].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i+1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print(
            f'EP_{epoch}, avg_loss={avg_loss / len(data_iter)}, total_acc={total_correct / total_element * 100}')

    def save(self, epoch, file_path="output/bert_trained.model"):
        output_path = file_path + f'.ep{epoch}'
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print(f'EP:{epoch} Model Saved on: {output_path}')
        return output_path
