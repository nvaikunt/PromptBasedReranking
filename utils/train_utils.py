import datasets
import torch
from transformers import Seq2SeqTrainer
from tqdm import tqdm


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
        current_inputs = [f"Question: {question} Passage: {text[0]} Relevant: " for text in texts]
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


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        batch_size = labels.size(dim=0)

        log_softmax = torch.nn.LogSoftmax(dim=-1)

        outputs = model(**inputs)
        logits = outputs.get("logits")
        outputs = log_softmax(logits)

        pos_end = batch_size // 2
        ce_loss = torch.nn.CrossEntropyLoss()

        pos_outputs = outputs[:pos_end, :, :]
        neg_outputs = outputs[pos_end:, :, :]
        flat_size = pos_outputs.size(-1)

        pos_loss = ce_loss(pos_outputs.view(-1, flat_size), labels[:pos_end, :].view(-1))
        neg_loss = ce_loss(neg_outputs.view(-1, flat_size), labels[pos_end:, :].view(-1))

        margin = 1
        margin_loss = torch.nn.MarginRankingLoss(margin)
        loss = margin_loss(pos_loss, neg_loss, torch.tensor(-1))

        return (loss, outputs) if return_outputs else loss
