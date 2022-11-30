import torch
import datasets


def create_evidence_texts(data_filepath: str, delimiter="\t", file_type="csv") -> datasets.Dataset:
    return datasets.load_dataset(file_type, data_files=data_filepath, split="train", delimiter=delimiter)


def get_top_k_pos(row, k: int, txt_database: datasets.Dataset) -> list:
    ctxs = row["ctxs"]
    top_k = []
    for ctx in ctxs:
        if ctx["has_answer"]:
            text = txt_database[ctx["id"] - 1]["text"]
            top_k.append((text, "true"))
        if len(top_k) == k:
            break
    if len(top_k) == 0:
        return []
    while len(top_k) < k:
        top_k.extend(top_k[:(k - len(top_k))])
    return top_k[:k]


def get_top_k_pos_neg(row, k: int, txt_database: datasets.Dataset):
    ctxs = row["ctxs"]
    top_k_pos = []
    top_k_neg = []
    for ctx in ctxs:
        if ctx["has_answer"] and len(top_k_pos) < k:
            text = txt_database[ctx["id"] - 1]["text"]
            top_k_pos.append((text, "true"))
        if not ctx["has_answer"] and len(top_k_neg) < k:
            text = txt_database[ctx["id"] - 1]["text"]
            top_k_neg.append((text, "false"))
        if len(top_k_pos) == k and len(top_k_neg):
            break
    if len(top_k_pos) == 0:
        return []
    while len(top_k_pos) < k:
        top_k_pos.extend(top_k_pos[:(k - len(top_k_pos))])
    while len(top_k_neg) < k:
        top_k_neg.extend(top_k_neg[:(k - len(top_k_neg))])
    top_k = []
    top_k.extend(top_k_pos[:k])
    top_k.extend(top_k_neg[:k])
    return top_k


def create_pos_txt_col(example, k: int, txt_database: datasets.Dataset) -> dict:
    return {"pos_text": get_top_k_pos(example, k, txt_database)}


def create_pos_neg_txt_col(example, k: int, txt_database: datasets.Dataset) -> dict:
    return {"pos_neg_text": get_top_k_pos_neg(example, k, txt_database)}


def preprocess_function(examples, tokenizer,
                        max_input_length: int, max_target_length: int, input_col: str) -> dict:
    model_inputs = tokenizer(
        examples[input_col],
        max_length=max_input_length,
        truncation=True,
        padding="longest")
    labels = tokenizer(text_target=examples["targets"], max_length=max_target_length, truncation=True,
                       padding="longest", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs


def qg_batching(question: str, ctxs: list,
                evidence_txts: datasets.Dataset) -> datasets.Dataset:
    texts = [evidence_txts[ctx["id"] - 1]["text"] for ctx in ctxs]
    texts = [f"Passage: {text} Please write a question based on this passage" for text in texts]
    targets = [question for text in texts]
    eval_dataset = datasets.Dataset.from_dict({'inputs': texts, 'targets': targets})
    return eval_dataset


def relevance_batching(question: str, ctxs: list, has_ans: torch.BoolTensor,
                       evidence_txts: datasets.Dataset) -> datasets.Dataset:
    texts = [evidence_txts[ctx["id"] - 1]["text"] for ctx in ctxs]
    # Below is real baseline, but commenting for now to minimize current train / test imbalance
    # texts = [f"Query: {question} Document: {text} Relevant: " for text in texts]
    # Below is the temp current line to match incorrect training
    texts = [f"Question: {question} Passage: {text} Relevant: " for text in texts]
    targets = ["true" if ans else "false" for ans in has_ans]
    new_dataset = datasets.Dataset.from_dict({'inputs': texts, 'targets': targets})
    return new_dataset


def qg_ranking(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    log_soft = log_softmax(logits)
    labels = labels.unsqueeze(2)
    log_soft = log_soft.gather(2, labels).squeeze(2)
    log_soft = log_soft.mean(dim=1)
    return log_soft


def relevance_ranking(logits: torch.Tensor, truth_ix) -> torch.Tensor:
    softmax = torch.nn.Softmax(dim=-1)
    probs = softmax(logits)
    probs = probs[:, 0, truth_ix]
    return probs
