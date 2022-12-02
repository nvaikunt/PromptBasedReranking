import argparse
from preprocess_data import create_training_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import torch.nn as nn
from utils.train_utils import CustomTrainer, parse_train_args
_truth_ix = 1176


class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True) -> torch.Tensor:
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, :-self.n_tokens])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([input_embedding, learned_embedding], 1)


def prompt_train(args: argparse.Namespace):

    model_ckpt, train_dir, valid_dir, evidence_dir = args.model_ckpt, \
                                                     args.train_dir, args.val_dir, args.evidence_dir

    model = T5ForConditionalGeneration.from_pretrained("google/t5-base-lm-adapt")
    tokenizer = AutoTokenizer.from_pretrained("google/t5-base-lm-adapt")
    n_tokens = int(args.n_tokens)
    soft_embed = SoftEmbedding(model.get_input_embeddings(), n_tokens,
                               initialize_from_vocab=True)
    model.set_input_embeddings(soft_embed)

    batch_size = int(args.batch_size)
    grad_accum_steps = int(args.grad_accumulation_steps)
    warm_up_steps = int(args.warmup_steps)

    num_epochs = int(args.num_epochs)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)

    if args.isQG == "True":
        isQG = True
    else:
        isQG = False
    if args.isRanking == "True":
        isRanking = True
    else:
        isRanking = False
    train_data_sz = int(args.max_train_size)
    validation_data_sz = int(args.max_val_size)

    train_dataset, _ = create_training_dataset(train_dir, evidence_dir, train_data_sz,
                                               isQG, isRanking, batch_size, tokenizer,
                                               args.dataset_verbose, is_prompt=True, n_tokens=n_tokens)
    validation_dataset, _ = create_training_dataset(valid_dir, evidence_dir, validation_data_sz,
                                                    isQG, isRanking, batch_size, tokenizer,
                                                    args.dataset_verbose, is_prompt=True, n_tokens=n_tokens)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length")

    output_dir = args.outdir

    for name, param in model.named_parameters():
        if name != "shared.learned_embedding":
            param.requires_grad = False

    if args.push == "True":
        push = True
    else:
        push = False
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        gradient_accumulation_steps=grad_accum_steps,
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        warmup_steps=warm_up_steps,
        save_steps=5000,
        eval_steps=5000,
        logging_strategy="steps",
        logging_steps=1000,
        num_train_epochs=num_epochs,
        save_total_limit=4,
        hub_token=args.hub_token,
        push_to_hub=push,
        load_best_model_at_end=True
    )

    if not isRanking:
        trainer = Seq2SeqTrainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
    else:
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
    trainer.train()
    if push:
        trainer.push_to_hub(commit_message=f'{args.run_name}')
    if args.do_eval:
        trainer.evaluate()



def main(args):
    prompt_train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    parser.add_argument("--n_tokens", type=str, required=False, default="100",
                        help="Number of Soft Prompt Tokens")
    arguments = parser.parse_args()
    main(arguments)
