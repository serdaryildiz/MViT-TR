import torch

from dataset.charMapper import CharMapper
import editdistance as ed

mapper = CharMapper()


def getAcc(preds: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor):
    assert preds.size() == targets.size()
    B, _ = preds.size()
    assert len(lengths) == B

    correctChar = 0
    totalChar = 0
    correctWord = 0

    for i in range(B):
        l = lengths[i]
        p = preds[i][:l]
        t = targets[i][:l]

        corr: torch.tensor = p == t
        correctChar += torch.sum(corr[:-1].to(torch.int))
        correctWord += int(torch.all(corr))
        totalChar += l - 1

    wordAcc = correctWord / B
    charAcc = correctChar / totalChar
    return wordAcc, charAcc


if __name__ == '__main__':
    targets = torch.randint(41, (4, 25))
    logits = torch.rand((4, 25, 41))
    lengths = torch.tensor([2, 3, 5, 7])
    preds = torch.max(logits, dim=2)[1]
    getAcc(preds, targets, lengths)
