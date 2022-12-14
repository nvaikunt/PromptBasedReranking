import datasets
from datasets import Dataset
import torch
from transformers import Seq2SeqTrainer
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
from transformers.utils.import_utils import is_datasets_available
from typing import Optional


def create_ranking_loss_baseline_examples(dataset: datasets.Dataset, n: int = None) -> dict:
    if not n:
        n = len(dataset)
    with_answer = 0
    inputs = []
    targets = []
    for i in tqdm(range(n)):
        texts = dataset[i]["pos_neg_text"]
        if not texts:
            continue
        question = dataset[i]["question"]
        current_inputs = [f"Query: {question} Document: {text[0]} Relevant: " for text in texts]
        current_targets = [text[1] for text in texts]
        inputs.extend(current_inputs)
        targets.extend(current_targets)
        with_answer += 1
    k = [len(targets) / (with_answer * 2)] * len(targets)
    return {"inputs": inputs, "targets": targets, "k_pos_neg": k}


def create_q_gen_baseline_examples(dataset: datasets.Dataset, n: int = None) -> dict:
    if not n:
        n = len(dataset)
    inputs = []
    targets = []
    with_answer = 0
    for i in tqdm(range(n)):
        texts = dataset[i]["pos_text"]
        if not texts:
            continue
        question = dataset[i]["question"]
        current_inputs = [f"Passage: {text[0]} Please write a question based on this passage" for text in texts]
        current_targets = [question for text in texts]
        inputs.extend(current_inputs)
        targets.extend(current_targets)
        with_answer += 1
    k = [len(targets) / with_answer] * len(targets)
    return {"inputs": inputs, "targets": targets, "k_pos": k}


def create_q_gen_ranking_baseline_examples(dataset: datasets.Dataset, n: int = None) -> dict:
    if not n:
        n = len(dataset)
    inputs = []
    targets = []
    with_answer = 0
    for i in tqdm(range(n)):
        texts = dataset[i]["pos_neg_text"]
        if not texts: continue
        question = dataset[i]["question"]
        current_inputs = [f"Passage: {text[0]} Please write a question based on this passage" for text in texts]
        current_targets = [question for text in texts]
        inputs.extend(current_inputs)
        targets.extend(current_targets)
        with_answer += 1
    k = [len(targets) / (with_answer * 2)] * len(targets)
    return {"inputs": inputs, "targets": targets, "k_pos_neg": k}


def ranking_loss(logits, labels, margin=1, isQG=True, truth_ix=1176):

    batch_size = labels.size(dim=0)

    pos_end = batch_size // 2
    ce_loss = torch.nn.CrossEntropyLoss()

    pos_outputs = logits[:pos_end, :, :]
    neg_outputs = logits[pos_end:, :, :]
    flat_size = pos_outputs.size(-1)

    if isQG:
        pos_loss = ce_loss(pos_outputs.view(-1, flat_size), labels[:pos_end, :].view(-1))
        neg_loss = ce_loss(neg_outputs.view(-1, flat_size), labels[pos_end:, :].view(-1))
    else:
        pos_outputs = pos_outputs[:, 0, :]
        neg_outputs = neg_outputs[:, 0, :]
        true_ids = torch.ones(pos_end) * truth_ix
        pos_loss = ce_loss(pos_outputs.view(-1, flat_size), true_ids)
        neg_loss = ce_loss(neg_outputs.view(-1, flat_size), true_ids)

    margin_loss = torch.nn.MarginRankingLoss(margin)
    loss = margin_loss(pos_loss, neg_loss, torch.tensor(-1))
    return loss


def parse_train_args(parser):

    parser.add_argument("--isQG", type=str, required=False, default="True",
                        help="Indicates whether model will be trained with Question Generation objective")
    parser.add_argument("--isRanking", type=str, required=False, default="False",
                        help="Indicates whether Ranking loss or Cross Entropy loss is being used")
    parser.add_argument("-ep", "--num_epochs", type=str, required=True, default=3,
                        help="Number of Epochs to Train for")
    parser.add_argument("-bs", "--batch_size", type=str, required=False, default=10, help="Batch Size")
    parser.add_argument("-lr", "--learning_rate", type=str, required=False, default=5e-4, help="Learning Rate")
    parser.add_argument("-trsz", "--max_train_size", type=str, required=False, default=80000,
                        help="Max number of questions to be considered in training set")
    parser.add_argument("-vlsz", "--max_val_size", type=str, required=False, default=9000,
                        help="Max number of questions to be considered in validation set")
    parser.add_argument("-odir", "--outdir", type=str, required=True, default="baseline-qg-ce",
                        help="Directory where Training Outputs will be saved to")
    parser.add_argument("-edir", "--evidence_dir", type=str, required=True,
                        help="Directory where Evidence Data will be saved to")
    parser.add_argument("-tdir", "--train_dir", type=str, required=True,
                        help="Directory where Train Data will be saved to")
    parser.add_argument("-vdir", "--val_dir", type=str, required=True,
                        help="Directory where Validation Data will be saved to")
    parser.add_argument("-m", "--model_ckpt", type=str, required=False, default="google/t5-base-lm-adapt",
                        help="Checkpoint to start training from")
    parser.add_argument("--do_eval", action='store_true',
                        help="Evaluated at the end of training")
    parser.add_argument("--dataset_verbose", action='store_true',
                        help="Print Progress Bars for Dataset Map function")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Run Name")
    parser.add_argument("--hub_token", type=str, required=False, default="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",
                        help="Token ID")
    parser.add_argument("--push", type=str, required=False, default="True",
                        help="To PUSH to HUB or NOT")
    parser.add_argument("--grad_accumulation_steps", type=str, required=True, default="1",
                        help="Number of Iterations to Perform Before Update, Effective Batch_sz is "
                             "Batch SZ * Grad Step")
    parser.add_argument("--warmup_steps", type=str, required=False, default="1000",
                        help="Number of intital steps before hitting max lr")
    parser.add_argument("--weight_decay", type=str, required=False, default="5e-5",
                        help="Decoupled Weight Regularizer")
    parser.add_argument("--strategy", type=str, required=False, default="steps",
                        help="Logging and Saving Strategy")
    return parser

class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, margin=1,
                     return_outputs=False, isQG=True, truth_ix=1176):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = ranking_loss(logits, labels, margin, isQG, truth_ix)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

#        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

#        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

