import datasets
from datasets.utils import disable_progress_bar
from functools import partial
from transformers import PreTrainedTokenizerBase
from utils.data_utils import create_evidence_texts, create_pos_txt_col, create_pos_neg_txt_col, preprocess_function, \
    preprocess_func_soft_prompt
from utils.train_utils import create_q_gen_baseline_examples, create_ranking_loss_baseline_examples, \
    create_q_gen_ranking_baseline_examples


def create_training_dataset(data_filepath: str, evidence_filepath: str, data_sz: int,
                            isQG: bool, isRanking: bool,
                            batch_sz: int, tokenizer: PreTrainedTokenizerBase,
                            map_verbose: bool, is_prompt: bool = False, n_tokens=100):
    if not map_verbose:
        disable_progress_bar()
    evidence_txt = create_evidence_texts(evidence_filepath)
    full_dataset = datasets.load_dataset("json", data_files=data_filepath, split="train",
                                         cache_dir='/projects/tir5/users/nvaikunt/')
    full_dataset = full_dataset.map(partial(create_pos_txt_col, k=20,
                                            txt_database=evidence_txt), num_proc=4)
    full_dataset = full_dataset.map(partial(create_pos_neg_txt_col, k=(20 // 2),
                                            txt_database=evidence_txt), num_proc=4)
    if not isQG:
        train_creation_fct = create_ranking_loss_baseline_examples
        drop_col = "k_pos_neg"
    elif isQG and isRanking:
        train_creation_fct = create_q_gen_ranking_baseline_examples
        drop_col = "k_pos_neg"
    else:
        train_creation_fct = create_q_gen_baseline_examples
        drop_col = "k_pos"
    if data_sz > len(full_dataset):
        data_sz = len(full_dataset)
    train_dict = train_creation_fct(full_dataset, n=data_sz)
    train_dataset = datasets.Dataset.from_dict(train_dict)
    if is_prompt:
        train_dataset = train_dataset.map(partial(preprocess_func_soft_prompt, tokenizer=tokenizer, n_tokens=n_tokens,
                                                  max_input_length=412, max_target_length=50), batched=True)
    else:
        train_dataset = train_dataset.map(partial(preprocess_function, tokenizer=tokenizer,
                                                  max_input_length=512, max_target_length=50), batched=True)

    train_dataset = train_dataset.remove_columns(["inputs", "targets", drop_col])
    train_dataset.set_format(type="torch")
    return train_dataset, evidence_txt


def create_eval_dataset(data_filepath: str, evidence_filepath: str, data_sz: int,
                            map_verbose: bool):
    if not map_verbose:
        disable_progress_bar()
    evidence_txt = create_evidence_texts(evidence_filepath)
    full_dataset = datasets.load_dataset("json", data_files=data_filepath, split="train")
    if data_sz > len(full_dataset):
        data_sz = len(full_dataset)
    eval_dataset = full_dataset.select(range(data_sz))
    return eval_dataset, evidence_txt



