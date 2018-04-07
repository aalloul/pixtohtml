from re import sub
from string import punctuation
from random import choices
from os.path import isfile


def identify_digits(ch):
    return ch.isdigit()


def identify_punctuation(ch):
    return ch in punctuation


def identify_newline(ch):
    return ch in ["\n", "\r", "\cr"]


def identify_atsymbo(ch):
    return ch == "@"


def identifiers(ch):
    return identify_digits(ch) or identify_newline(ch) or \
           identify_punctuation(ch) or identify_atsymbo(ch)


def remove_char(line):
    return "".join(["" if identifiers(char) else char for char in line])


def remove_line_id(line):
    return " ".join(line.split(" ")[1:])


def store_result(words):
    with open("../data/corpus.txt", "w") as f:
        for w in words.split(" "):
            f.write(w + "\n")


def get_corpus(infile="../data/text.txt", stored=True, nwords=5000):

    if stored and isfile("../data/corpus.txt"):
        with open("../data/corpus.txt", "r") as f:
            return f.read().split("\n")

    cnt = 0
    out_lines = ""
    with open(infile, "r") as f:
        for line in f:
            # Skip header
            cnt += 1
            if cnt < 4:
                continue

            li = remove_line_id(line)
            li = remove_char(li)
            li = sub("\s{2,", " ", li)
            out_lines += li

    out_lines = [word for word in out_lines.split(" ") if word != ""]
    corpus = choices(out_lines, k=nwords)

    with open("../data/corpus.txt", "w") as f:
        [f.write(word+"\n") for word in corpus]

    return corpus


if __name__ == "__main__":
    print(get_corpus(infile="../data/text.txt", nwords=50))
