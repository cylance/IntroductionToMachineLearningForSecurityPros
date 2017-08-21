import numpy as np
import os
import re
import h5py
import socket
import struct
from sklearn.preprocessing import normalize


LOG_REGEX = re.compile(r'([^\s]+)\s[^\s]+\s[^\s]+\s\[[^\]]+\]\s"([^\s]*)\s[^"]*"\s([0-9]+)')


def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]


def get_prevectors():
    data_path = "data/www.secrepo.com/self.logs/"
    # ensure we get the IPs used in the examples
    prevectors = {
        ip2int("192.187.126.162"): {"requests": {}, "responses": {}},
        ip2int("49.50.76.8"): {"requests": {}, "responses": {}},
        ip2int("70.32.104.50"): {"requests": {}, "responses": {}},
    }
    for path in os.listdir(data_path):
        full_path = os.path.join(data_path, path)
        with open(full_path, "r") as f:
            for line in f:
                try:
                    ip, request_type, response_code = LOG_REGEX.findall(line)[0]
                    ip = ip2int(ip)
                except IndexError:
                    continue

                if ip not in prevectors:
                    if len(prevectors) >= 10000:
                        continue
                    prevectors[ip] = {"requests": {}, "responses": {}}

                if request_type not in prevectors[ip]["requests"]:
                    prevectors[ip]['requests'][request_type] = 0

                prevectors[ip]['requests'][request_type] += 1

                if response_code not in prevectors[ip]["responses"]:
                    prevectors[ip]["responses"][response_code] = 0

                prevectors[ip]["responses"][response_code] += 1

    return prevectors


def convert_prevectors_to_vectors(prevectors):
    request_types = [
        "GET",
        "POST",
        "HEAD",
        "OPTIONS",
        "PUT",
        "TRACE"
    ]
    response_codes = [
        200,
        404,
        403,
        304,
        301,
        206,
        418,
        416,
        403,
        405,
        503,
        500,
    ]

    vectors = np.zeros((len(prevectors.keys()), len(request_types) + len(response_codes)), dtype=np.float32)
    ips = []

    for index, (k, v) in enumerate(prevectors.items()):
        ips.append(k)
        for ri, r in enumerate(request_types):
            if r in v["requests"]:
                vectors[index, ri] = v["requests"][r]
        for ri, r in enumerate(response_codes):
            if r in v["responses"]:
                vectors[index, len(request_types) + ri] = v["requests"][r]

    return ips, vectors


if __name__ == "__main__":
    prevectors = get_prevectors()
    ips, vectors = convert_prevectors_to_vectors(prevectors)
    vectors = normalize(vectors)

    with h5py.File("secrepo.h5", "w") as f:
        f.create_dataset("vectors", shape=vectors.shape, data=vectors)
        f.create_dataset("cluster", shape=(vectors.shape[0],), data=np.zeros((vectors.shape[0],), dtype=np.int32))
        f.create_dataset("notes", shape=(vectors.shape[0],), data=np.array(ips))

    print "Finished prebuilding samples"
