import argparse
from preprocess_data import create_training_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.train_utils import CustomTrainer, parse_train_args
_truth_ix = 1176


def train(args: argparse.Namespace):
    model_ckpt, train_dir, valid_dir, evidence_dir = args.model_ckpt, \
                                                     args.train_dir, args.val_dir, args.evidence_dir
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = T5ForConditionalGeneration.from_pretrained(model_ckpt)

    batch_size = int(args.batch_size)
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
                                               args.dataset_verbose)
    validation_dataset, _ = create_training_dataset(valid_dir, evidence_dir, validation_data_sz,
                                                    isQG, isRanking, batch_size, tokenizer,
                                                    args.dataset_verbose)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length")

    num_epochs = int(args.num_epochs)
    learning_rate = float(args.learning_rate)
    grad_accum_steps = int(args.grad_accumulation_steps)
    warm_up_steps = int(args.warmup_steps)
    weight_decay = float(args.weight_decay)
    output_dir = args.outdir
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
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    arguments = parser.parse_args()
    main(arguments)
