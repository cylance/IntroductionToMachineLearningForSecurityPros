from gevent.monkey import patch_all
patch_all()
from gevent.pool import Pool
import json

from idpanel.training.prevectorization import load_all_panel_urls, load_all_panel_paths
from idpanel.utility import make_request
from os.path import isfile


def get_item_from_site((panel_name, base_url, path)):
    url = base_url + path
    for attempt in xrange(3):
        try:
            code, content = make_request(url)
            return json.dumps(
                {
                    "code": code,
                    "content_ssdeep": content,
                    "url": url,
                    "base_url": base_url,
                    "offset": path,
                    "label": panel_name,
                }) + "\n"
        except Exception, e:
            print e
            continue
    return None


if __name__ == "__main__":
    data_points = set()
    if isfile("prevectors.json"):
        with open("prevectors.json", "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                try:
                    line = json.loads(line)
                except:
                    print line
                    raise
                data_points.add(line["url"])

    requests_to_make = []
    paths = load_all_panel_paths()
    urls = load_all_panel_urls()
    for path in paths:
        for url in urls:
            if url[1] + path not in data_points:
                requests_to_make.append((url[0], url[1], path))

    print "Making", len(requests_to_make), "requests"
    pool = Pool(size=10)
    completed = 0
    with open("prevectors.json", "a") as f:
        for result in pool.imap_unordered(get_item_from_site, requests_to_make):
            completed += 1
            print "{0} completed out of {1}".format(completed, len(requests_to_make))
            if result is not None:
                f.write(result)