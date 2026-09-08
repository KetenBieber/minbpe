import tiktoken

from hw_2 import RegexTokenizer

""" 不重新训练GPT4-Tokenizer,直接使用tiktoken的mergeable_ranks
    我们多余做一步：把tiktoken的最终token表反推出历史mergerule表

    GPT4-Tokenizer是一个预训练tokenizer，所以不允许用户进行train

"""


class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        # 注册GPT4官方的分词正则表达式
        pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        super().__init__(pattern)
        # 不重新训练，直接用cl100k_base的merge
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.mergeable_ranks = self.enc._mergeable_ranks
        self.byte_shuffle = {}
        self.inverse_byte_shuffle = {}
        for i in range(256):
            # raw byte id -> GPT base token id
            self.byte_shuffle[i] = self.mergeable_ranks[bytes([i])]
            self.inverse_byte_shuffle[self.mergeable_ranks[bytes([i])]] = i

        # 从gpt4官方tokenizer merges中恢复出分词规则
        self._recover_merges()
        # 重建vocab
        self._build_vocab()

        # 按照gpt4标准注册special tokens
        GPT4_SPECIAL_TOKENS = {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        }
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _recover(self, token: bytes, max_rank: int) -> list[bytes]:
        parts = [bytes([b]) for b in token]

        while True:
            rank = max_rank
            # 直接对所有的pair进行计算
            for i in range(len(parts) - 1):
                # 如果有更小的rank，更新rank和idx
                if self.mergeable_ranks.get(parts[i] + parts[i + 1], max_rank) < rank:
                    rank = min(rank, self.mergeable_ranks.get(parts[i] + parts[i + 1]))
                    idx = i
            # 如果已经没有更小的rank,直接退出
            if rank == max_rank:
                break

            # rank合理 ，进行合并
            parts[idx : idx + 2] = [parts[idx] + parts[idx + 1]]

        return parts

    def _recover_merges(self):
        # 遍历mergeable_ranks中所有token,需要先让mergeable_rank按rank顺序排列
        for token, rank in sorted(
            self.mergeable_ranks.items(), key=lambda item: item[1]
        ):
            # 单byte token是基础token，而非merge生成的，直接跳过
            if len(token) == 1:
                continue
            else:
                # merge
                parts = self._recover(token, max_rank=rank)

                assert len(parts) == 2
                left, right = parts
                left_id = self.mergeable_ranks[left]
                right_id = self.mergeable_ranks[right]

                self.merges[(left_id, right_id)] = rank

    def _build_vocab(self):

        # 明确写成按idx排序
        for pair, idx in sorted(self.merges.items(), key=lambda item: item[1]):
            # for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def _encode_chunk(self, ids):
        # GPT-4 permutation
        shuffle_ids = []
        for item in ids:
            shuffle_ids.append(self.byte_shuffle[item])

        return super()._encode_chunk(shuffle_ids)

    def train(self, text, vocab_size):
        raise NotImplementedError("GPT4-Tokenizer not allowed to train!")

    def decode(self, ids: list[int]) -> str:
        # 添加GPT-4 inverse shuffle

        # vocab 是id -> bytes

        parts = []
        for token_id in ids:
            if token_id in self.vocab:
                # 普通token
                shuffle_bytes = self.vocab[token_id]
                raw_bytes = bytes(
                    self.inverse_byte_shuffle[shuffle_id]
                    for shuffle_id in shuffle_bytes
                )
                parts.append(raw_bytes)
            elif token_id in self.inverse_special_tokens:
                special_token = self.inverse_special_tokens[token_id]
                special_token = special_token.encode("utf-8")
                parts.append(special_token)
            else:
                raise ValueError("invalid tokenid")
        words = b"".join(parts)
        result = words.decode("utf-8")
        return result


def bpe(mergeable_ranks, token: bytes, max_rank: int) -> list[bytes]:
    parts = [bytes([b]) for b in token]

    while True:
        rank = max_rank
        # 直接对所有的pair进行计算
        for i in range(len(parts) - 1):
            # 如果有更小的rank，更新rank和idx
            if mergeable_ranks.get(parts[i] + parts[i + 1], max_rank) < rank:
                rank = min(rank, mergeable_ranks.get(parts[i] + parts[i + 1]))
                idx = i
        # 如果已经没有更小的rank,直接退出
        if rank == max_rank:
            break

        # rank合理 ，进行合并
        parts[idx : idx + 2] = [parts[idx] + parts[idx + 1]]

    return parts


def recover_merges(mergeable_ranks, base_vocab_size) -> dict[tuple[int, int], int]:
    merges = {}

    # 遍历mergeable_ranks中所有token,需要先让mergeable_rank按rank顺序排列
    for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1]):
        if rank < base_vocab_size:
            # skip
            continue
        else:
            # merge
            parts = bpe(mergeable_ranks, token, max_rank=rank)

            assert len(parts) == 2
            left, right = parts
            left_id = mergeable_ranks[left]
            right_id = mergeable_ranks[right]

            merges[(left_id, right_id)] = rank
    return merges


if __name__ == "__main__":

    # enc = tiktoken.get_encoding("cl100k_base")

    # # 证明UTF-8 byte value != cl100k_base的基础token_id, byte permutation/ byte shuffle
    # for text in [
    #     "a",
    #     "A",
    #     " ",
    #     "!",
    #     "hello",
    #     " hello",
    #     "你好",
    # ]:
    #     print(text, enc.encode(text))

    # mergeable_ranks = enc._mergeable_ranks

    # print(type(mergeable_ranks))
    # print(len(mergeable_ranks))
    # print(mergeable_ranks[b"!"])
    # print(mergeable_ranks[b"A"])
    # print(mergeable_ranks[b"a"])
    # print(mergeable_ranks[b" "])

    # # inverse byte shuffle

    # # byte shuffle
    # byte_shuffle = {}
    # inverse_byte_shuffle = {}
    # for i in range(256):
    #     # raw byte id -> GPT base token id
    #     byte_shuffle[i] = mergeable_ranks[bytes([i])]
    #     inverse_byte_shuffle[mergeable_ranks[bytes([i])]] = i

    # assert byte_shuffle[ord("!")] == 0
    # assert byte_shuffle[ord("A")] == 32
    # assert byte_shuffle[ord("a")] == 64
    # assert byte_shuffle[ord(" ")] == 220

    # assert inverse_byte_shuffle[64] == ord("a")

    # for i in range(256):
    #     assert inverse_byte_shuffle[byte_shuffle[i]] == i

    # pattern = r"\w+|\s+|[^\w\s]+"

    # tokenizer = GPT4Tokenizer()
    # assert tokenizer.encode("a", allowed_special="none") == [64]
    # assert tokenizer.encode("A", allowed_special="none") == [32]
    # assert tokenizer.encode("!", allowed_special="none") == [0]
    # assert tokenizer.encode(" ", allowed_special="none") == [220]

    # assert tokenizer.decode(tokenizer.encode("a", allowed_special="none")) == "a"

    # assert tokenizer.decode(tokenizer.encode("A", allowed_special="none")) == "A"

    # assert tokenizer.decode(tokenizer.encode("!", allowed_special="none")) == "!"

    # assert tokenizer.decode(tokenizer.encode(" ", allowed_special="none")) == " "

    # text = "你好，GPT!"

    # ids = tokenizer.encode(text, allowed_special="none")

    # print(ids)

    # assert tokenizer.decode(ids) == text

    # mergeable_ranks = {
    #     b"a": 0,
    #     b"b": 1,
    #     b"c": 2,
    #     b"ab": 3,
    #     b"abc": 4,
    # }

    # parts = bpe(
    #     mergeable_ranks,
    #     b"abc",
    #     max_rank=4,
    # )

    # print(parts)

    ours = GPT4Tokenizer()
    official = tiktoken.get_encoding("cl100k_base")

    print(ours.encode(" hello", allowed_special="none"))

    print(official.encode(" hello", allowed_special=set()))

    text = "hello<|endoftext|>world"

    print(ours.encode(text, allowed_special="all"))
    print(ours.encode(text, allowed_special="none"))
    # print(ours.encode(text, allowed_special="none_raise"))

    test_texts = [
        "hello",
        " hello",
        "hello world!",
        "hello123!!!",
        "你好",
        "你好，世界！",
        "hello 你好 world!",
        "123456789",
        "\nhello\nworld\n",
    ]

    for text in test_texts:
        ours_ids = ours.encode(text, allowed_special="none")

        official_ids = official.encode(text, allowed_special=set())

        print(text)
        print("ours:    ", ours_ids)
        print("official:", official_ids)

        assert ours_ids == official_ids

    texts = [
        "hello",
        " hello",
        "你好，世界！",
        "hello 你好 world!",
        "123456789",
        "\nhello\nworld\n",
    ]

    for text in texts:
        assert ours.decode(ours.encode(text, allowed_special="none")) == text

    text = "hello<|endoftext|>world"

    ids = ours.encode(text, allowed_special="all")

    assert ours.decode(ids) == text
