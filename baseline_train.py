import argparse
from preprocess_data import create_training_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.train_utils import CustomTrainer
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
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    num_epochs = int(args.num_epochs)
    learning_rate = float(args.learning_rate)
    grad_accum_steps = int(args.grad_accumulation_steps)
    output_dir = args.outdir
    if args.push == "True":
        push = True
    else: 
        push = False	
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        gradient_accumulation_steps=grad_accum_steps,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=.05,
        save_steps=10000,
        logging_strategy="epoch",
        num_train_epochs=num_epochs,
        hub_token=args.hub_token,
        push_to_hub=push
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
    parser.add_argument("--isQG", type=str, required=False, default="True",
                        help="Indicates whether model will be trained with Question Generation objective")
    parser.add_argument("--isRanking", type=str, required=False, default="False",
                        help="Indicates whether Ranking loss or Cross Entropy loss is being used")
    parser.add_argument("-ep", "--num_epochs", type=str, required=True, default=3,
                        help="Number of Epochs to Train for")
    parser.add_argument("-bs", "--batch_size", type=str, required=False, default=10, help="Batch Size")
    parser.add_argument("-lr", "--learning_rate", type=str, required=False, default=5e-4, help="Learning Rate")
    parser.add_argument("-ngpu", "--num_gpu", type=str, required=False, default=1, help="number of GPUs to be used")
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
    parser.add_argument("--grad_accumulation_steps", type=str, required=False, default="1",
                        help="To PUSH to HUB or NOT")
    arguments = parser.parse_args()
    main(arguments)
