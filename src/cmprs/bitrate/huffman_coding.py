from heapq import heappush, heappop, heapify

import torch

from cmprs.transform import Transform, FOut


def huffman_bitrate(probs):
    # min heap - the lowest combined probability at the top
    # each heap node contains a list of per-integer probabilities
    heap = [(p, [p]) for p in probs if p > 0]

    # build the heap
    heapify(heap)

    # iteratively combine two heap nodes with the lowest frequencies
    bitrate = 0
    while len(heap) > 1:
        key_1, probs_1 = heappop(heap)
        key_2, probs_2 = heappop(heap)

        # We add together the integer probabilities each time we would otherwise add a bit
        # to their representing string
        bitrate += sum(probs_1)
        bitrate += sum(probs_2)

        heappush(heap, (key_1 + key_2, probs_1 + probs_2))

    return bitrate


class HuffmanCodingBitrate(Transform):

    def forward(self, x) -> FOut:
        if x.is_floating_point():
            x_int = x.to(torch.int64)
        else:
            x_int = x
        probs = (torch.bincount(x_int) / x.numel()).cpu().numpy()

        bitrate = huffman_bitrate(probs)

        return FOut(tx=x, bitrate=bitrate)


huffmanCodingBitrate = HuffmanCodingBitrate()
