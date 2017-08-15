import os


def load_all_panel_paths():
    paths = set()
    for path in os.listdir("panel_paths"):
        path = os.path.join("panel_paths", path)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith("./"):
                    line = line[2:]
                paths.add(line)

    return list(paths)


def load_panel_paths(panel):
    paths = set()
    path = os.path.join("panel_paths", panel + ".txt")
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("./"):
                line = line[2:]
            paths.add(line)

    return list(paths)


def load_all_panel_urls():
    panels = []
    for path in os.listdir("c2_labels"):
        panel_name = path.split(".")[0]
        path = os.path.join("c2_labels", path)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[-1] != "/":
                    line += "/"
                panels.append((panel_name, line))

    return panels