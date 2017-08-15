import requests
import ssdeep


def make_request(url, quiet=False, raw_results=False):
    if not quiet:
        print "Requesting {0}".format(url)
    r = requests.get(url, allow_redirects=False, timeout=90)
    content = r.content
    return r.status_code, ssdeep.hash(content) if not raw_results else content.encode('hex')
