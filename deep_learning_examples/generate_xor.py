import numpy as np
import os


def generate_random_string(length):
    return os.urandom(length)


def principal_period(s):
    i = (s+s).find(s, 1, -1)
    return None if i == -1 else s[:i]


def make_dataset(dataset, nsamp, slen, maxkeylen):
    x = np.zeros((nsamp, slen, 8))
    y = np.zeros((nsamp, maxkeylen))

    for i in xrange(nsamp):
        keylen = np.random.randint(maxkeylen) + 1

        # save key len as categorical variable
        y[i, keylen - 1] = 1.0

        dataptr = np.random.randint(len(dataset) - slen)
        data = dataset[dataptr:dataptr + slen]
        data = np.fromstring(data, dtype=np.uint8)

        key = generate_random_string(keylen)
        while principal_period(key) is not None:
            key = generate_random_string(keylen)

        key = np.fromstring(key, dtype=np.uint8)

        key_nrep = int(np.ceil(float(slen) / float(len(key))))
        key_exp = np.tile(key, key_nrep)[:slen]

        xor_ciphertext = np.bitwise_xor(data, key_exp)

        x[i, :, :] = np.unpackbits(xor_ciphertext).reshape(slen, 8)

    return x, y


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog=__file__,
        description="Generates XOR'd data",
    )
    parser.add_argument('-n', '--bytes-in-key', type=int, required=True, help='Number of bytes in XOR key')
    parser.add_argument('-o', '--output', required=True, help="File to write encrypted data to")
    args = parser.parse_args()

    print "Loading data"
    with open("enwik8", "r") as f:
        dataset = f.read()

    dataptr = np.random.randint(len(dataset) - 64)
    data = dataset[dataptr:dataptr + 64]

    print "Plaintext:", repr(data)
    print "Key size:", args.bytes_in_key

    key = generate_random_string(args.bytes_in_key)
    while principal_period(key) is not None:
        key = generate_random_string(args.bytes_in_key)

    print "Key:", repr(key)
    cipher_text = ""

    for i in xrange(len(data)):
        cipher_text += chr(ord(data[i]) ^ ord(key[i % len(key)]))

    print "Cipher text:", repr(cipher_text)

    with open(args.output, "wb") as f:
        f.write(cipher_text)
