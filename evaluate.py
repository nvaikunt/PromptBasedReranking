import torch
from tqdm import tqdm
from transformers import T5Config,DataCollatorForSeq2Seq, AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from functools import partial
from utils.data_utils import qg_batching, relevance_batching, qg_ranking, relevance_ranking, \
    preprocess_function, preprocess_func_soft_prompt
import datasets
from utils.train_utils import ranking_loss
import numpy as np
import argparse
from preprocess_data import create_eval_dataset
from prompt_tuning_train import SoftEmbedding


def evaluate_recall(validation, k, model, tokenizer, batch_size, evidence_txts,
                    is_prompt, truth_ix, n_tokens, isRanking=False, isQG=True):
    assert k % batch_size == 0, "k must be multiple of batch_size"
    assert batch_size % 2 == 0, "Batch Size Must Be Even"

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
            eval_dataset = qg_batching(question, ctxs, evidence_txts)
        else:
            eval_dataset = relevance_batching(question, ctxs, has_ans, evidence_txts)

        datasets.utils.disable_progress_bar()
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        if is_prompt:
            eval_dataset = eval_dataset.map(partial(preprocess_func_soft_prompt, tokenizer=tokenizer,
                                            max_input_length=412, n_tokens=n_tokens,
                                            max_target_length=50),
                                        batched=True)
        else:
            eval_dataset = eval_dataset.map(partial(preprocess_function, tokenizer=tokenizer,
                                            max_input_length=512,
                                             max_target_length=50),
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
                    score = relevance_ranking(logits, truth_ix)
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


def print_eval_stats(filepath, run_name, dpr_recall, rerank_recall, loss):
    with open(filepath, "w") as f:
        f.write(f"Eval Loss for {run_name} is {loss}")
        f.write("\n")
        for i, (orig_recall, new_recall) in enumerate(zip(dpr_recall, rerank_recall)):
            f.write(f"Recall@{i + 1} for DPR: {orig_recall}")
            f.write("\n")
            f.write(f"Recall@{i + 1} for {run_name}: {new_recall}")
            f.write("\n")
        f.write(f"Finish Stats for Run {run_name}")
        f.write("\n")


def main(args: argparse.Namespace):
    model_ckpt, eval_data, outfile, evidence_dir = args.model_ckpt, \
                                                     args.eval_data, args.outfile, args.evidence_dir
    batch_sz = int(args.batch_size)
    k = int(args.k)
    max_eval_size = int(args.max_eval_size)
    n_tokens = int(args.n_tokens)
    if args.is_prompt == "True":
        is_prompt = True

    else:
        is_prompt = False
    if args.is_prompt:
        init_config = T5Config.from_pretrained(model_ckpt)
        model = T5ForConditionalGeneration(init_config)
        soft_embed = SoftEmbedding(model.get_input_embeddings(), n_tokens,
                                   initialize_from_vocab=True)
        model.set_input_embeddings(soft_embed)
        model = model.from_pretrained(model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if args.isQG == "True":
        isQG = True
    else:
        isQG = False
    if args.isRanking == "True":
        isRanking = True
    else:
        isRanking = False


    validation_dataset, evidence_txt = create_eval_dataset(eval_data, evidence_dir, max_eval_size,
                                                           args.dataset_verbose)

    base_recall, exp_recall, eval_loss = evaluate_recall(validation_dataset, k, model, tokenizer,
                                                         batch_sz, evidence_txt, is_prompt, 1176,
                                                         n_tokens, isRanking, isQG)
    print_eval_stats(outfile, args.run_name, base_recall, exp_recall, eval_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--isQG", type=str, required=False, default="True",
                        help="Indicates whether model will be trained with Question Generation objective")
    parser.add_argument("--isRanking", type=str, required=False, default="False",
                        help="Indicates whether Ranking loss or Cross Entropy loss is being used")
    parser.add_argument("--max_eval_size", type=str, required=False, default=9000,
                        help="Max number of questions to be considered in validation set")
    parser.add_argument( "--outfile", type=str, required=True, default="output.txt",
                        help="Path to output file")
    parser.add_argument("--evidence_dir", type=str, required=True,
                        help="Path where Evidence Data is kept")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path where Evaluation Dataset is Kept")
    parser.add_argument("-m", "--model_ckpt", type=str, required=True,
                        help="Checkpoint to start eval from")
    parser.add_argument("--batch_size", type=str, required=False, default=10,
                        help="Eval Batch Size")
    parser.add_argument("--is_prompt", type=str, required=True, default="False",
                        help="Model is Prompt Tuning Model")
    parser.add_argument("--n_tokens", type=str, required=False, default="100",
                        help="Number of Prompt Tokens, only needed if prompt")
    parser.add_argument("--dataset_verbose", action='store_true',
                        help="Print Progress Bars for Dataset Map function")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Run Name")
    parser.add_argument("--k", type=str, required=True, default=20, help="Number of Contexts to be considered")
    arguments = parser.parse_args()
    main(arguments)
