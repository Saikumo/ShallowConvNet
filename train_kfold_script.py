import argparse
from train_kfold import train_kfold

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=int)
parser.add_argument("--gpu", type=int)
args = parser.parse_args()

best_epoch = train_kfold(
    subjectId=args.subject,
    device=f'cuda:{args.gpu}'
)

print(f"Subject {args.subject}: best epoch = {best_epoch}")

with open(f"results/subj_{args.subject}.txt", "w") as f:
    f.write(str(best_epoch))
