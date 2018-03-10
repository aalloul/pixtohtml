from requests import get
from json import dump
import logging
from sys import stdout
import pypandoc
import imgkit
from json import load
from glob import glob
from time import time
from re import sub

# Logging
logging.basicConfig(stream=stdout, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GithubCrawler(object):

    def __init__(self):
        self.url = "https://api.github.com/"
        self.token = "e524a1168eb34ddc846edd58bc2deaa7c938268f"
        self.list_repos = []

    @staticmethod
    def _format_link(link: str) -> str:
        """
        Expects link from the header in the format
        '<url>; rel="next", <https://api.github.com/repositories{?since}>;
        rel="first"'

        :param link: String following the format outlined above. The URL is
        expected to contain the parameters `since` and the `access_token`
        :return: The URL itself
        """
        return link.split(";")[0].replace("<", "").replace(">", "")

    def get_batch(self, link=None):
        start = time()
        if link is None:
            logger.debug("First batch started")
            url = self.url + "repositories?access_token=" + self.token
        else:
            logger.debug("Follow-up batch started")
            url = link

        g = get(url)
        if 199 < g.status_code < 300:
            logger.debug("batch download took {}".format(time() - start))
            return g.json(), self._format_link(g.headers["link"])
        else:
            raise ConnectionError("Status code = {}, reason = {}".format(
                g.status_code, g.reason
            ))

    def list_public_repos(self, n=10000, from_disk=False):
        logger.info("  Looking for n = {} repositories".format(n))
        cnt = 0

        if from_disk:
            logger.info("Reading repos list from disk")
            with open("../data/public_repo/list_public_repos.json", "r") as f:
                self.list_repos = load(f)
        else:
            try:
                self.list_repos, link_ = self.get_batch()
                cnt += len(self.list_repos)
                logger.debug("cnt = {}".format(cnt))

                while cnt < n:
                    tmp_, link_ = self.get_batch(link_)
                    self.list_repos += tmp_
                    cnt += len(tmp_)

                    if len(self.list_repos) >= 400:
                        logger.info("  Done ~ {}".format(cnt))
                        self.store_list_repos(cnt//400)
                        self.list_repos[:] = []

                if len(self.list_repos) > 0:
                    self.store_list_repos((cnt//400)+1)

            except ConnectionError:
                self.store_list_repos((cnt//400)+1)

    def store_list_repos(self, iterator,
                         fname="../data/public_repo/list_public_repos_{}.json"):
        with open(fname.format(iterator), "w") as f:
            dump(self.list_repos, f, indent=4)

    @staticmethod
    def get_list_repos():
        list_files = glob("../data/public_repo/list_public_repos*.json")
        out_ = []

        for file in list_files:
            with open(file, "r") as f:
                out_ += [j['full_name'] for j in load(f)]

        return out_


def get_readme(repo_name, store_function):
    url = 'https://raw.githubusercontent.com/' \
          '{}/master/README.md'.format(repo_name)

    req = get(url)
    if 200 <= req.status_code < 300:
        return store_function(strip_out_images(req.text), repo_name)
    elif req.status_code == 404 and req.reason == 'Not Found':
        return ""
    else:
        logger.error("Could not download readme from {}.\n"
                     "Status = {}, Reason = {}, ".format(
                        repo_name, req.status_code, req.reason))
        return ""


def split_pages(text: str, lines_per_page=50):
    out = []
    cnt = 0
    out_ = ""

    for line in text.split("\n"):
        cnt += 1
        out_ += line+"\n"
        if cnt == lines_per_page:
            out.append(out_)
            out_ = ""
            cnt = 0
    return out


def store_readme(text, repo_name):
    """
    Do not store readme files with less than 20 characters
    :param text:
    :param repo_name:
    :return:
    """
    if len(text.replace("#", "")) < 20:
        return

    out_ = split_pages(text)
    cnt = 0
    fnames = []
    for o in out_:
        cnt += 1
        filename = repo_name.replace("/", "_") + "{}.md".format(cnt)
        with open("../data/markdown/{}".format(filename), "w") as f:
            f.write(o)
        fnames.append(filename)

    return fnames


def generate_html(filename):
    outfile_name = filename.replace(".md", ".html")
    outfile = "../data/html/{}".format(outfile_name)
    infile = "../data/markdown/{}".format(filename)
    pypandoc.convert_file(infile, 'html', outputfile=outfile,
                          extra_args=['--standalone'])
    return outfile_name


def generate_jpeg(filename):
    outfile_name = filename.replace(".html", ".jpeg")
    outfile = "../data/jpeg/{}".format(outfile_name)
    infile = "../data/html/{}".format(filename)

    try:
        imgkit.from_file(infile, outfile)
    except Exception as ex:
        logger.warning("Exception caught for the file {}.\n"
                       "Exception = {}".format(filename, ex))

    return outfile_name

def strip_out_images(text):
    # [stuff](http://travis-ci.org/3scale/3scale_ws_api_for_dotnet)
    return sub("\[.+\]\(http.+\)", "", text)



if __name__ == "__main__":
    # gcrawler = GithubCrawler()
    # logger.info("Get list of public repositories")
    # gcrawler.list_public_repos(40000, from_disk=False)
    # list_repos = gcrawler.get_list_repos()
    #
    # logger.info("Download markdown version")
    # markdown_files = [get_readme(repo, store_readme) for repo in list_repos]
    # list_repos[:] = []
    # logger.info("Stored {} Readme files".format(len(markdown_files)))
    #
    # # Flatten & Remove empty slots
    # out_ = []
    # for m in markdown_files:
    #     if m == '':
    #         continue
    #     else:
    #         out_ += m
    #
    # markdown_files = [o for o in out_]
    # out_[:] = []

    markdown_files = glob("../data/markdown/*md")
    logger.info("Generate HTML")
    html_files = [generate_html(m.split("/")[-1]) for m in markdown_files]
    markdown_files[:] = []

    logger.info("Generate JPEG")
    jpeg_file = [generate_jpeg(h) for h in html_files]
    jpeg_file[:] = []
    logger.info("Done")
