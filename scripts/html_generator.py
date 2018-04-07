from random import choices, randint
import string
import imgkit
import logging
from sys import stdout
from scripts.corpus_generator import get_corpus

# Logging
logging.basicConfig(stream=stdout, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def process_dag(dag, corpus):
    header = """<!DOCTYPE html>
    <html>"""

    title = "<head>\n<title>{text}</title>\n</head>\n".format(
        text=generate_random_text(corpus, 3, 10)
    )
    body = "<body>\n"

    for tag in dag:
        if tag == "ul":
            body += "<ul><li>{text}</li>\n".format(
                text=generate_random_text(corpus, 3, 20))

            for _ in range(0, randint(0, 5)):
                body += "<li>{text}</li>\n".format(
                    text=generate_random_text(corpus, 3, 20))
        else:
            body += "<{tag}> {text} </{tag}>\n".format(
                tag=tag, text=generate_random_text(corpus, 3, 40)
            )

    body += "</body>"
    footer = """</html>"""

    html_for_jpeg = header+"\n"+title+"\n"+body+"\n"+footer
    html_for_training = header+"\n"+"\n"+body+"\n"+footer
    return html_for_jpeg, html_for_training


def generate_random_text(corpus, min_chars=1, max_chars=1000):
    alpha_num = corpus + [d for d in string.digits]

    return " ".join(choices(alpha_num, k=randint(min_chars, max_chars)))


def generate_html(html_tags, corpus):
    tag_names = [t["tag"] for t in html_tags]
    tag_weights = [t["weight"] for t in html_tags]
    dag = choices(tag_names, weights=tag_weights, k=randint(3, 10))
    return process_dag(dag, corpus)


def generate_jpeg(filename, **kwargs):
    outfile = "../data/jpeg/{}.jpeg".format(filename)
    infile = "../data/html_jpeg/{}.html".format(filename)

    try:
        imgkit.from_file(infile, outfile, **kwargs)
    except Exception as ex:
        logger.warning("Exception caught for the file {}.\n"
                       "Exception = {}".format(filename, ex))

if __name__ == "__main__":
    tags = [
        {"tag": "p", "weight": 27273},
        {"tag": "h1", "weight": 1820},
        {"tag": "h2", "weight": 4064},
        {"tag": "h3", "weight": 2349},
        {"tag": "ul", "weight": 2674}
    ]
    corpus = get_corpus(stored=True, nwords=200)
    html_generator = (generate_html(tags, corpus) for _ in range(5000))

    cnt = 0
    for html_to_jpeg, html_to_train in html_generator:
        cnt += 1
        print("File number {}".format(cnt))

        with open("../data/html_train/{}.html".format(cnt), "w") as f:
            f.write(html_to_train)

        with open("../data/html_jpeg/{}.html".format(cnt), "w") as f:
            f.write(html_to_jpeg)

        generate_jpeg(cnt)

