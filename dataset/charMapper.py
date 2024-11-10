import re
import torch


class CharMapper:
    lower2upper = {
        ord(u"i"): u"İ",
        ord(u"ı"): u"I"
    }

    upper2lower = {
        ord(u"İ"): u"i",
        ord(u"I"): u"ı"
    }

    def __init__(self, letters: str = "0123456789abcçdefgğhıijklmnoöpqrsştuüvwxyz", maxLength: int = 25):
        self.letters = letters
        self.maxLength = maxLength
        self.map = {"[END]": 0}
        self.reverseMap = {0: "[END]"}
        i = 1
        for l in self.letters:
            self.map[l] = i
            self.reverseMap[i] = l
            i += 1
        self.map["[PAD]"] = i
        self.reverseMap[i] = "[PAD]"
        return

    def __call__(self, text: str, return_length=False):
        text = self.text2label(text)
        length = len(text) + 1
        mappedText = torch.tensor([self.map[l] for l in text] + [self.map["[END]"]])
        text = torch.ones((self.maxLength + 1,)) * self.map["[PAD]"]
        text[:len(mappedText)] = mappedText
        if return_length:
            return text, length
        else:
            return text

    def reverseMapper(self, label: torch.tensor):
        label = label.cpu()
        text = "".join([self.reverseMap[l] for l in label.numpy()])
        return text.split("[END]")[0]

    def text2label(self, text):
        text = re.sub('[^0-9a-zA-ZğüşöçıİĞÜŞÖÇ]+', '', text)
        text = text.translate(self.upper2lower).lower()
        return text


if __name__ == '__main__':
    mapper = CharMapper()
    mapped = mapper("!MA-PİŞ$Z")
    print(mapped)
