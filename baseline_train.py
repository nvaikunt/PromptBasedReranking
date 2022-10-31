import argparse
from preprocess_data import create_training_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
_truth_ix = 1176


def train(args: argparse.Namespace):
    model_ckpt, train_dir, valid_dir, evidence_dir = args.m, args.tdir, args.vdir, args.edir
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM(model_ckpt)

    batch_size = args.bs
    isQG = args.isQG
    isRanking = args.isRanking
    train_data_sz = args.trsz
    validation_data_sz = args.vlsz

    train_dataset = create_training_dataset(train_dir, evidence_dir, train_data_sz,
                                            isQG, isRanking, batch_size, tokenizer)
    validation_dataset = create_training_dataset(valid_dir, evidence_dir, validation_data_sz,
                                                 isQG, isRanking, batch_size, tokenizer)


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--isQG", action='store_false',
                        help="Indicates whether model will be trained with Question Generation objective")
    parser.add_argument("--isRanking", action='store_true',
                        help="Indicates whether Ranking loss or Cross Entropy loss is being used")
    parser.add_argument("-ep", "--num_epochs", type=int, required=True, default=3,
                        help="Number of Epochs to Train for")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, default=10, help="Batch Size")
    parser.add_argument("-lr", "--learning_rate", type=float, required=True, default=5e-4, help="Learning Rate")
    parser.add_argument("-ngpu", "--num_gpu", type=int, required=True, default=1, help="number of GPUs to be used")
    parser.add_argument("-trsz", "--max_train_size", type=int, required=True, default=90000,
                        help="Max number of questions to be considered in training set")
    parser.add_argument("-vlsz", "--max_val_size", type=int, required=True, default=4000,
                        help="Max number of questions to be considered in validation set")
    parser.add_argument("-odir", "--outdir", type=str, required=True, default="baseline-qg-ce",
                        help="Directory where Training Outputs will be saved to")
    parser.add_argument("-edir", "--evidence_dir", type=str, required=True,
                        help="Directory where Evidence Data will be saved to")
    parser.add_argument("-tdir", "--train_dir", type=str, required=True,
                        help="Directory where Train Data will be saved to")
    parser.add_argument("-vdir", "--val_dir", type=str, required=True,
                        help="Directory where Validation Data will be saved to")
    parser.add_argument("-m", "--model_ckpt", type=str, required=False,
                        help="Checkpoint to start training from")
    arguments = parser.parse_args()
    main(arguments)
