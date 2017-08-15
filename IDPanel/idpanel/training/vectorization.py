import json
import ssdeep
import numpy as np
from scipy.sparse import lil_matrix


def load_data_from_results_file(path):
    c2_bases = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            data = json.loads(line)
            data['content'] = data['content'].decode('hex')
            data['content_ssdeep'] = ssdeep.hash(data['content'])

            if data['base_url'] not in c2_bases:
                c2_bases[data['base_url']] = {}
            data["offset"] = data["url"][len(data["base_url"]):]
            print "{0}  -  {1}  -  {2}".format(data['code'], data['base_url'], data['offset'])
            c2_bases[data['base_url']][data['offset']] = data

    return c2_bases


def vectorize(feature_set, c2_data):
    vector = np.zeros((len(feature_set),), dtype=np.float)
    for index, (offset, code, ssdeep_hash) in enumerate(feature_set):
        if offset not in c2_data:
            continue
        if c2_data[offset]["code"] == code:
            d = ssdeep.compare(c2_data[offset]["content_ssdeep"], ssdeep_hash)
            d = float(d) / float(100.0)
            vector[index] = d

    return vector


def vectorize_with_sparse_features(sparse_feature_set, feature_count, c2_data):
    vector = lil_matrix((1, feature_count), dtype=np.float)
    for index, (offset, code, ssdeep_hash) in sparse_feature_set:
        if offset not in c2_data:
            continue
        if c2_data[offset]["code"] == code:
            d = ssdeep.compare(c2_data[offset]["content_ssdeep"], ssdeep_hash)
            d = float(d) / float(100.0)
            vector[0, index] = d

    return vector


def psuedo_vector_entries(c2_bases):

    # identify all keys
    all_keys = set()
    for c2 in c2_bases.keys():
        all_keys |= set(c2_bases[c2].keys())

    # reduce to only the set that was queried for all C2s
    common_keys = set()
    for key in all_keys:
        common = True
        for c2 in c2_bases.keys():
            if key not in c2_bases[c2]:
                common = False
                break
        if common:
            common_keys.add(key)

    # Build a feature set from these results
    # Feature format (offset, code/ssdeep, value)
    features = []
    for key in common_keys:
        codes = []
        sss = []

        for c2 in c2_bases.keys():
            codes.append(c2_bases[c2][key]["code"])
            sss.append(c2_bases[c2][key]["content_ssdeep"])

        for code in set(codes):
            features.append((key, "code", code))

        for ss in set(sss):
            features.append((key, "ssdeep", ss))

    vectors = []
    c2s = c2_bases.keys()
    for c2 in c2s:
        vectors.append((c2, vectorize(features, c2_bases[c2]).tolist()))

    return features, vectors


def load_raw_feature_vectors():
    with open("raw_feature_vectors.json", "r") as f:
        d = json.load(f)
        return d["labels"], d["names"], d["vectors"]


def load_c2_vectors(bot_name):
    with open("c2_vectors/{0}_vectors.json".format(bot_name), "r") as f:
        data = json.load(f)

    return data['vectors'], data['features']