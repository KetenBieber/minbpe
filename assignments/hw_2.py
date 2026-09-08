import regex as re

"""
实现RegexTokenizer
"""


class RegexTokenizer:
    """RegexTokenizer实现"""

    def __init__(self, pattern: str):
        # 初始化为byte-level
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        # 保留分词规则（正则表达式）
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)

        # 保留decode所需
        self.special_tokens = {}  # str->int
        self.inverse_special_tokens = {}  # int->str

    def _split(self, text: str) -> list[str]:
        """按照 regex pattern 划分文本"""
        return self.compiled_pattern.findall(text)

    def _split_special_tokens(
        self, text: str, allowed_special_tokens: dict[str, int]
    ) -> list[str]:
        """针对special tokens进行分词"""

        # 如果没有special tokens
        if not allowed_special_tokens:
            return [text]

        # escape将special tokens转义
        pattern = "|".join(re.escape(token) for token in allowed_special_tokens)

        pattern = f"({pattern})"

        text = re.split(pattern, text)

        return text

    @staticmethod
    def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        """统计词频"""
        counts = {}

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def _pre_tokenization(self, text: str) -> list[list[int]]:
        """预分词:分词+转码"""

        # 分割成词
        words = self._split(text)
        ids = []

        for word in words:
            ids.append(list(word.encode("utf-8")))
        return ids

    @staticmethod
    def _chunk_stats(text_ids: list[list[int]]) -> dict[tuple[int, int], int]:
        # 拿到分好的词
        counts = {}
        for tmp_ids in text_ids:
            # 统计词频
            tmp_count = RegexTokenizer._get_stats(tmp_ids)

            for key in tmp_count:
                counts[key] = counts.get(key, 0) + tmp_count.get(key)

        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """单独ids的merge操作"""
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

    @staticmethod
    def _chunk_merge(
        ids: list[list[int]], pair: tuple[int, int], idx: int
    ) -> list[list[int]]:
        """merge操作"""

        new_ids = []
        for item_ids in ids:
            # 逐个ids进行merge
            tmp_ids = RegexTokenizer._merge(item_ids, pair, idx)
            new_ids.append(tmp_ids)
        # 返回
        return new_ids

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        # 清空一下原有的special tokens & inverse tokens
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        for key, value in special_tokens.items():
            self.special_tokens[key] = value
            self.inverse_special_tokens[value] = key

    def train(self, text: str, vocab_size: int):
        """输入文本进行bpe训练"""
        assert vocab_size >= 256

        # 清空上次训练状态
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # 预分词
        ids = self._pre_tokenization(text)

        num_merge = vocab_size - 256

        # 进行num_merge次的merge
        for i in range(num_merge):
            # 统计词频
            stats = self._chunk_stats(ids)

            # 如果当前已经不存在pair
            if not stats:
                break

            # 最大词频的pair
            pair = max(stats, key=stats.get)

            # 新idx
            idx = 256 + i

            # 进行merge
            ids = self._chunk_merge(ids, pair, idx)

            # 记录merge
            self.merges[pair] = idx

        # merge完毕，build vocab
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        """设计职责：对单个chunk做encode"""
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            results = {}

            for key in self.merges:
                if stats.get(key) is None:
                    continue
                results[key] = self.merges[key]

            # 如果已没有可合并的
            if not results:
                break

            pair = min(results, key=results.get)

            idx = results.get(pair)

            ids = self._merge(ids, pair, idx)

        return ids

    def encode(self, text: str, allowed_special: str | set[str]) -> list[int]:
        """text-> chunk encode -> flatten -> list[int]"""

        # allowed_special: "all" | "none" | "none_raise" | set[str]

        # 本次编码中需要用到的special token
        allowed_special_tokens = {}

        if isinstance(allowed_special, str):
            if allowed_special == "all":
                # 直接用内置的字典
                allowed_special_tokens = dict(self.special_tokens)
            elif allowed_special == "none":
                # 注册过的special token全部换成普通文本编码
                pass
            elif allowed_special == "none_raise":
                # 不允许special token，只要原文本里出现任意已注册的special token，直接报错

                # 直接检查文本里有没有special token
                for token in self.special_tokens:
                    if token in text:
                        raise ValueError("none raise mode: exist special token!")

            else:
                raise ValueError("allowed_special must be 'all'、'none' 或 set[str]")
        elif isinstance(allowed_special, set):
            # 传入的是一个允许的special token集合
            for token in allowed_special:
                if token not in self.special_tokens:
                    raise ValueError(
                        "Could not find a special token in registered special token"
                    )
                else:
                    allowed_special_tokens[token] = self.special_tokens[token]
        else:
            raise TypeError("allowed_special must be str or set[str]")

        # 根据新的special token拆分special tokens 和 text
        segments = self._split_special_tokens(text, allowed_special_tokens)

        result = []
        for segment in segments:
            # 检查是否为当前encode指定special tokens中的special token
            if segment in allowed_special_tokens:
                # 是
                result.append(allowed_special_tokens[segment])  # 直接加进编码
                continue
            else:
                # 不是

                # 对special token前的语句进行分词
                ids = self._pre_tokenization(segment)
                for item_ids in ids:
                    # 逐个chunk进行encode
                    item_ids = self._encode_chunk(item_ids)
                    for i in item_ids:
                        result.append(i)

        return result

    def decode(self, ids: list[int]) -> str:
        parts = []
        for token_id in ids:
            if token_id in self.vocab:
                parts.append(self.vocab[token_id])
            elif token_id in self.inverse_special_tokens:
                sepecial_token = self.inverse_special_tokens[token_id]
                sepecial_token = sepecial_token.encode("utf-8")
                parts.append(sepecial_token)
            else:
                raise ValueError("invalid tokenid")
        words = b"".join(parts)
        result = words.decode("utf-8")
        return result


if __name__ == "__main__":
    # \w+ 匹配连续word; \s+匹配连续空格符号
    pattern = r"\w+|\s+|[^\w\s]+"

    tokenizer = RegexTokenizer(pattern)
    print(tokenizer._split("hello world!"))
    print(tokenizer._split("hello   world!!!"))
    print(tokenizer._split("hello123 world"))
    print(tokenizer._split("你好，world!"))

    chunk_ids = [
        [1, 2, 1, 2],
        [1, 2, 3],
    ]

    ids = tokenizer._chunk_stats(chunk_ids)
    # print(tokenizer._chunk_merge(ids, ))

    pattern = r"\w+|\s+|[^\w\s]+"

    tokenizer = RegexTokenizer(pattern)

    tokenizer.train("hello hello hello!", 265)

    print(tokenizer.merges)

    for idx in range(256, len(tokenizer.vocab)):
        print(idx, tokenizer.vocab[idx])

    tokenizer.train("hello hello hello!", 265)

    text = "hello world!"

    ids = tokenizer.encode(text, "all")

    print(ids)
    print(type(ids))

    decoded = tokenizer.decode(ids)

    assert decoded == text

    tokenizer.register_special_tokens(
        {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
        }
    )
    assert tokenizer.special_tokens["<|endoftext|>"] == 100257
    assert tokenizer.inverse_special_tokens[100257] == "<|endoftext|>"

    text = "hello<|endoftext|>world"

    print(tokenizer._split_special_tokens(text, "all"))

    tokenizer.register_special_tokens({"<|endoftext|>": 100257})

    print(tokenizer.encode("hello<|endoftext|>world", "all"))

    text = "hello<|endoftext|>world"

    ids = tokenizer.encode(text, "all")

    print(ids)
    print(tokenizer.decode(ids))

    assert tokenizer.decode(ids) == text
    text = "hello<|endoftext|><|endoftext|>world"

    assert tokenizer.decode(tokenizer.encode(text, "all")) == text

    tokenizer.register_special_tokens(
        {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
        }
    )
    ids = tokenizer.encode("a<|endoftext|>b<|fim_prefix|>c", allowed_special="all")
    print(ids)
    ids = tokenizer.encode(
        "a<|endoftext|>b<|fim_prefix|>c", allowed_special={"<|endoftext|>"}
    )
    print(ids)

    tokenizer.encode("hello world", allowed_special="none_raise")  # 正常
    # tokenizer.encode("<|endoftext|>", allowed_special="none_raise")  # raise
    # tokenizer.encode("hello<|endoftext|>world", allowed_special="none_raise") # raise
    ids = tokenizer.encode("hello<|endoftext|>world", allowed_special="none")

    assert 100257 not in ids
