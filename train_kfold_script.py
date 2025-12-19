import argparse
from train_kfold import train_kfold
import torch
import os


def train_kfold_script(local_rank):
    if (local_rank == 0):
        subjects = [1, 3, 5, 7, 9]
        device = torch.device('cuda:0')
    else:
        subjects = [2, 4, 6, 8]
        device = torch.device('cuda:1')

    print(f"device {device}")

    for subject in subjects:

        print(f"test subject {subject}")

        best_epoch = train_kfold(
            subjectId=subject,
            device=device
        )

        print(f"Subject {subject}: best epoch = {best_epoch}")

        os.makedirs("results", exist_ok=True)  # 如果目录不存在就创建
        with open(f"results/subj_{subject}.txt", "w") as f:
            f.write(str(best_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, required=True)
    args = parser.parse_args()

    train_kfold_script(args.local_rank)
