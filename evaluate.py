import torch
import tdqm
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from functools import partial
from utils.data_utils import qg_batching, relevance_batching, qg_ranking, relevance_ranking
import datasets
from utils.train_utils import ranking_loss
import numpy as np


def evaluate_recall(validation, k, model, tokenizer, batch_size, evidence_txts,
                    preprocess_function, truth_ix, isRanking=False, isQG=True):
    assert k // batch_size != 0, "k must be multiple of batch_size"
    assert batch_size // 2 != 0, "Batch Size Must Be Even"

    if k < batch_size:
        batch_size = k

    original_recall = []
    current_recall = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    losses = []
    for i in tqdm(range(len(validation))):

        # Extract Question, Passages, and Info on Whether Passages have Answer
        question = validation[i]["question"]
        ctxs = validation[i]["ctxs"][:k]
        has_ans = [ctx["has_answer"] for ctx in ctxs]
        has_ans = torch.BoolTensor(has_ans)

        # Build Data as Model Expects
        if isQG:
            eval_dataset = qg_batching(question, ctxs, has_ans, evidence_txts)
        else:
            eval_dataset = relevance_batching(question, ctxs, has_ans, evidence_txts)

        datasets.utils.disable_progress_bar()
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        eval_dataset = eval_dataset.map(partial(preprocess_function, max_input_length=300,
                                                max_target_length=50, input_col='inputs'),
                                        batched=True)

        eval_dataset = eval_dataset.remove_columns(["inputs", "targets"])
        eval_dataset.set_format(type="torch")
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

        # Calculate Log Scores and Get Ranking
        scores = []
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                labels = batch["labels"]
                if isRanking:
                    loss = ranking_loss(logits, batch["labels"], 1,
                                        batch["labels"].size(dim=0))
                    losses.append(loss)
                else:
                    losses.append(outputs.loss)
                if isQG:
                    score = qg_ranking(logits, labels)
                else:
                    score = relevance_ranking(logits, labels, truth_ix)
                scores.append(score)

        scores = torch.cat(scores)
        topk_scores, indexes = torch.topk(scores, k=len(scores))

        # Collect Stats for Recall
        ranked_answers = has_ans[indexes]
        current_has_ans = torch.cumsum(ranked_answers, dim=0) > 0
        original_has_ans = torch.cumsum(has_ans, dim=0) > 0

        original_recall.append(original_has_ans.tolist())
        current_recall.append(current_has_ans.tolist())

    original_recall = np.mean(np.array(original_recall), axis=0)
    current_recall = np.mean(np.array(current_recall), axis=0)
    loss = sum(losses) / len(losses)
    return original_recall, current_recall, loss

