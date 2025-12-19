import argparse
from train_kfold import train_kfold


def train_kfold_script(local_rank):
    if (local_rank == 0):
        subjects = [1, 3, 5, 7, 9]
        device = 'cuda:0'
    else:
        subjects = [2, 4, 6, 8]
        device = 'cuda:1'

    for subject in subjects:
        best_epoch = train_kfold(
            subjectId=subject,
            device=device
        )

        print(f"Subject {subject}: best epoch = {best_epoch}")

        with open(f"results/subj_{subject}.txt", "w") as f:
            f.write(str(best_epoch))
