import datasets
from functools import partial
from transformers import PreTrainedTokenizerBase
from utils.data_utils import create_evidence_texts, create_pos_txt_col, create_pos_neg_txt_col, preprocess_function
from utils.train_utils import create_q_gen_baseline_examples, create_ranking_loss_baseline_examples, \
    create_q_gen_ranking_baseline_examples


def create_training_dataset(data_filepath: str, evidence_filepath: str, data_sz: int, isQG: bool, isRanking:bool,
                            batch_sz: int, tokenizer: PreTrainedTokenizerBase) -> datasets.Dataset:
    evidence_txt = create_evidence_texts(evidence_filepath)
    full_dataset = datasets.load_dataset("json", data_files=data_filepath, split="train")
    full_dataset = full_dataset.map(partial(create_pos_txt_col, k=batch_sz,
                                            txt_database=evidence_txt), num_proc=4)
    full_dataset = full_dataset.map(partial(create_pos_neg_txt_col, k=(batch_sz // 2),
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
    train_dataset = train_dataset.map(partial(preprocess_function, tokenizer, max_input_length=300,
                                              max_target_length=50, input_col='inputs'), batched=True)
    train_dataset = train_dataset.remove_columns(["inputs", "targets", drop_col])
    train_dataset.set_format(type="torch")
    return train_dataset






