from collections import defaultdict

"""
  实现最基础的BPE Tokenizer:BasicTokenizer
"""


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """统计词频"""

    # 初始化字典
    counts = {}
    # TODO:
    # 遍历 ids 中所有相邻元素
    # pair = (?, ?)
    #
    # 如果 pair 已经存在：
    #     次数 +1
    # 否则：
    #     初始化为 1

    for index in range(len(ids) - 1):
        pair = (ids[index], ids[index + 1])
        # 如果pair是第一次
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """合并高频token"""
    new_ids = []

    i = 0
    # 不一定固定步进，使用while
    while i < len(ids):
        # 检查是否最后一个
        if i + 1 == len(ids):
            new_ids.append(ids[i])
            i += 1
        else:
            # 检查当前pair
            tmp_pair = (ids[i], ids[i + 1])
            if tmp_pair == pair:
                # 如果是目标pair
                new_ids.append(idx)
                i += 2
            else:
                # 如果不是目标pair
                new_ids.append(ids[i])
                i += 1

    return new_ids


# num_merges = vocab_size - 256, num_merges就是我们要新增的词汇表数，这个多少，就意味着我们需要统计多少次词频+merge


def train_bpe(text: str, vocab_size: int):
    assert vocab_size >= 256

    # str -> bytes -> ids
    ids = list(text.encode("utf-8"))

    # 2. 要训练多少次 merge
    num_merges = vocab_size - 256

    # 保存训练结果
    merges = {}

    for i in range(num_merges):
        # 统计一次pair词频
        stats = get_stats(ids)

        # 找出本次最多的pair,使用max对dict操作时，需要指定使用key还是value操作
        more_pair = max(stats, key=stats.get)

        # 新的idx
        idx = 256 + i

        # merge
        ids = merge(ids, more_pair, idx)

        # 保存这个merge rule，也就是merge的参数
        merges[more_pair] = idx
        print(f"merge {i + 1}: " f"{more_pair} -> {idx}, " f"count={stats[more_pair]}")

    return merges, ids


def build_vocab(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
    """传入bpe训练的参数，得到一个vocab(可以从ids直接检索到token)
    依赖merge rule的训练顺序 ———— 后面的token可以依赖前面的token，但不能反过来
    """
    vocab = {}

    # [0 - 255]是字母表
    for i in range(256):
        vocab[i] = bytes([i])

    # merges训练的rules
    for pair, idx in merges.items():
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    return vocab


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """解码，传入ids和vocab解码"""
    # 1. 根据 ids 查 vocab
    words = b""

    # 比较低效
    # for i in ids:
    #     # 2. 把所有 bytes 拼起来
    #     words += vocab[i]

    # 高效写法
    words = b"".join(vocab[i] for i in ids)

    # 3. utf-8 decode
    result = words.decode("utf-8")
    return result


# 设计encode时，如果有训练过的pair同时出现，怎么处理merge的priority呢？
# merges中的idx 其实就可以表示merge priority


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    # UTF-8 -> byte ids
    ids = list(text.encode("utf-8"))

    while len(ids) >= 2:
        # 当前有哪些pair
        stats = get_stats(ids)

        # 找出其中”训练过，且优先级最高的pair“
        # 从merges中和stats中的比较，用stats的key去检索merges的token_id，然后取最小的token_id
        # results里就是pair -> tokenid
        results = {}
        for key in merges:
            # 如果没有检索到，跳过
            if stats.get(key) is None:
                continue
            results[key] = merges.get(key)

        # print(results)

        # 如果字典为空，则没有训练过的pair了，encode结束了
        if not results:
            break

        # 该pair为训练过的，且优先级最高的
        pair = min(results, key=results.get)

        idx = merges.get(pair)

        # merge
        ids = merge(ids, pair, idx)

    # 返回结果
    return ids


class BasicTokenizer:
    """BPE Tokenizer"""

    def __init__(self):
        self.merges = {}
        # 无训练，也初始化为byte-level tokenizer
        self.vocab = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        """统计词频"""
        counts = {}

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """merge操作"""
        new_ids = []

        i = 0
        while i < len(ids):
            # 检查是否是最后一个
            if i + 1 == len(ids):
                new_ids.append(ids[i])
                i += 1
                break

            # 检查有无对应pair可合并
            if (ids[i], ids[i + 1]) == pair:
                # 有
                new_ids.append(idx)
                i += 2
            else:
                # 没有
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def train(self, text: str, vocab_size: int):
        """从byte-level tokenizer开始训练bpe"""
        assert vocab_size >= 256

        # 清除上次训练结果
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # str -> utf-8 bytes
        ids = list(text.encode("utf-8"))

        # 要训练多少次merge
        merge_num = vocab_size - 256

        for i in range(merge_num):
            stats = self._get_stats(ids)

            if not stats:
                print("no enough pair! bpe training exit")
                break

            # 取出词频最大的pair
            pair = max(stats, key=stats.get)

            # new idx
            idx = 256 + i

            # merge
            ids = self._merge(ids, pair, idx)

            # 保存merge
            self.merges[pair] = idx

        # 训练完毕，重建vocab
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = self._get_stats(ids)

            results = {}
            for key in self.merges:
                if stats.get(key) is None:
                    continue
                results[key] = self.merges[key]

            # 如果已经没有可合并的token
            if not results:
                break

            pair = min(results, key=results.get)

            idx = self.merges.get(pair)

            ids = self._merge(ids, pair, idx)

        return ids

    def decode(self, ids: list[int]) -> str:
        words = b"".join(self.vocab[i] for i in ids)

        result = words.decode("utf-8")
        return result


if __name__ == "__main__":
    ids = [1, 2, 3, 1, 2]
    print(get_stats(ids))

    text = "aaabdaaabac"
    ids = list(text.encode("utf-8"))

    stats = get_stats(ids)

    print(ids)
    print(stats)
    print(max(stats, key=stats.get))

    # print(merge([1, 2, 3, 1, 2], (1, 2), 4))
    assert merge([1, 2, 3, 1, 2], (1, 2), 4) == [4, 3, 4]
    assert merge([1, 1, 1], (1, 1), 2) == [2, 1]
    assert merge([1, 2, 3], (5, 6), 7) == [1, 2, 3]
    assert merge([], (1, 2), 3) == []

    merges, ids = train_bpe("aaabdaaabac", vocab_size=259)

    print(merges)
    print(ids)

    # 一个值为97的序列
    # print(bytes([97]))
    # 97个值为0的序列
    # print(bytes(97))

    merges = {
        (97, 97): 256,
        (256, 97): 257,
        (257, 98): 258,
    }

    vocab = build_vocab(merges)

    assert len(vocab) == 259

    assert vocab[97] == b"a"
    assert vocab[98] == b"b"

    assert vocab[256] == b"aa"
    assert vocab[257] == b"aaa"
    assert vocab[258] == b"aaab"

    text = "aaabdaaabac"

    merges, ids = train_bpe(text, 259)
    vocab = build_vocab(merges)

    decoded = decode(ids, vocab)

    print(ids)
    print(decoded)

    assert decoded == text

    text = "你好， BPE！"

    ids = list(text.encode("utf-8"))

    vocab = {i: bytes([i]) for i in range(256)}

    assert decode(ids, vocab) == text
    print(decode(ids, vocab))

    print(256 < float("inf"))
    print(999999999 < float("inf"))

    # print(
    #     text.encode("utf-8")
    # )  # b'\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c BPE\xef\xbc\x81'
    # print(
    #     list(text.encode("utf-8"))
    # )  # [228, 189, 160, 229, 165, 189, 239, 188, 140, 32, 66, 80, 69, 239, 188, 129]

    # training_text = "aaabdaaabac"

    # merges, train_ids = train_bpe(training_text, vocab_size=259)

    # ids = encode("aaabaaab", merges)

    # print(ids)
    # assert decode(encode("aaabaaab", merges), build_vocab(merges)) == "aaabaaab"

    # text = "hello world"

    # ids = encode(text, merges)
    # decoded = decode(ids, build_vocab(merges))

    # assert decoded == text

    tokenizer = BasicTokenizer()

    text = "aaabdaaabac"

    tokenizer.train(text, 259)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(ids)
    print(decoded)

    assert decoded == text
    text = "你好，世界！Hello BPE!"

    tokenizer = BasicTokenizer()
    tokenizer.train(text, 270)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(text)
    print(ids)
    print(decoded)

    assert decoded == text

    tokenizer = BasicTokenizer()

    text = "我还没有训练 tokenizer"

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text

    text = "你好你好你好你好"
    tokenizer.train(text, 265)

    for idx in range(256, len(tokenizer.vocab)):
        print(idx, tokenizer.vocab[idx])
