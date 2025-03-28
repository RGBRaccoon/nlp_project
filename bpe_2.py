import asyncio
import json
import multiprocessing
import re
import time
from typing import List

TRAIN_FILE_PATH = "pg100.txt"
MAX_VOCAB = 30000
ASYNC_MAX_ITERATION = 10000
MIN_PAIR_FREQ = 40
VOCAB_FILE_PATH = "vocab.txt"
INFER_FILE_PATH = "input_1.txt"
OUTPUT_FILE_PATH = "output.txt"


class Tokenizer:
    def __init__(self):
        self.max_vocab = MAX_VOCAB

        self.vocab_file_path = VOCAB_FILE_PATH
        self.train_file_path = TRAIN_FILE_PATH
        self.infer_file_path = INFER_FILE_PATH
        self.output_file_path = OUTPUT_FILE_PATH
        self.min_pair_freq = MIN_PAIR_FREQ
        self.max_iteration = ASYNC_MAX_ITERATION

        self.worker_num = 20
        self.vocab = self.get_vocab()

        self.summary = {}

    def get_vocab(self) -> List[str]:
        with open(self.vocab_file_path, "r", encoding="utf-8") as f:
            vocab = f.readlines()
            vocab = [i.strip() for i in vocab]
        return vocab

    def save_vocab(self, vocab: List):
        with open(self.vocab_file_path, "w", encoding="utf-8") as f:
            for i in vocab:
                f.write(i + "\n")

        self.vocab = self.get_vocab()

    def get_txt_data(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        return data

    def preprocess(self, text: str, remove_special_chars: bool = True) -> List[List[str]]:
        if remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = text.lower()

        def whitespace_tokenize(text: str) -> List[str]:
            # 여러 개의 공백을 하나의 공백으로 정규화
            text = " ".join(text.split())
            # 공백을 기준으로 토큰화
            tokens = text.split(" ")
            return tokens

        # 공백을 기준으로 토큰화
        text_list = whitespace_tokenize(text)
        # print(text_list)
        new_text_list = []
        for chunk in text_list:
            # 각 단어를 문자 단위로 분리
            chars = list(chunk)
            new_text_list.append(chars)

        return new_text_list

    async def get_pairs(self, train_data: List[List[str]]) -> dict:
        worker_num = max(1, multiprocessing.cpu_count() - 1)
        chunk_size = max(1000, len(train_data) // worker_num)
        divided_train_data = [train_data[i : i + chunk_size] for i in range(0, len(train_data), chunk_size)]

        async def process_chunk(chunk: List[List[str]]) -> dict:
            pairs = {}
            for word in chunk:
                for i in range(len(word) - 1):
                    pair = tuple(word[i : i + 2])
                    pairs[pair] = pairs.get(pair, 0) + 1
            return pairs

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_chunk(chunk)) for chunk in divided_train_data]

        all_pairs = {}
        for task in tasks:
            pairs = task.result()
            for pair, count in pairs.items():
                all_pairs[pair] = all_pairs.get(pair, 0) + count

        return all_pairs

    async def update_chunks(self, train_data: List[List[str]], best_pair: tuple, new_vocab: str) -> List[List[str]]:
        worker_num = 20
        chunk_size = len(train_data) // worker_num
        divided_train_data = [train_data[i : i + chunk_size] for i in range(0, len(train_data), chunk_size)]

        async def process_chunk(chunk: List[List[str]]) -> List[List[str]]:
            updated_chunk = []
            for word in chunk:
                new_word = []
                i = 0
                while i < len(word) - 1:
                    if tuple(word[i : i + 2]) == best_pair:
                        new_word.append(new_vocab)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                if i < len(word):
                    new_word.append(word[i])
                updated_chunk.append(new_word)
            return updated_chunk

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_chunk(chunk)) for chunk in divided_train_data]

        updated_data = []
        for task in tasks:
            chunk = task.result()
            updated_data.extend(chunk)

        return updated_data

    async def update_vocab(self, remove_special_chars: bool = True):
        time_start = time.time()
        train_data = self.get_txt_data(self.train_file_path)
        train_data: List[List[str]] = self.preprocess(train_data, remove_special_chars=remove_special_chars)
        iteration = 0

        while True:
            current_time = time.time()
            print(f"iteration : {iteration}/{self.max_iteration}, time : {current_time - time_start}")
            pairs = await self.get_pairs(train_data)
            if not pairs:
                stop_reason = "pairs is empty"
                break

            best_pair = max(pairs, key=pairs.get)
            # if pairs[best_pair] < self.min_pair_freq:  # 빈도수가 너무 낮으면 종료
            #     stop_reason = "stop by low frequency"
            #     break

            new_vocab = "".join(best_pair)
            train_data = await self.update_chunks(train_data, best_pair, new_vocab)

            iteration += 1
            if iteration >= self.max_iteration:
                stop_reason = "stop by max iteration"
                break

        # 빈도수 기준으로 정렬하여 상위 max_vocab개 선택
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
        vocab = ["".join(pair) for pair, _ in sorted_pairs[: self.max_vocab]]

        time_end = time.time()

        self.summary = {
            "실제 반복횟수": iteration,
            "stop_reason": stop_reason,
            "max vocab": self.max_vocab,
            "vocab 크기": len(vocab),
        }
        self.save_vocab(vocab)

    def tokenize(self):
        self.vocab = self.get_vocab()
        # 입력받은 텍스트 데이터를 vocab를 기반으로 토큰화
        start_time = time.time()
        input_data = self.get_txt_data(self.infer_file_path)
        input_data: List[str] = self.preprocess(input_data, remove_special_chars=False)

        # vocab 기반으로 토큰화

        result = []
        success_count = 0
        fail_count = 0

        progress = 0
        work = len(input_data)
        for word in input_data:
            progress += 1
            print(f"{progress}/{work}")
            word_str = "".join(word)
            tokens = []
            start = 0

            while start < len(word_str):
                end = len(word_str)
                found = False

                # 가장 긴 매칭을 찾기
                while end > start:
                    # print(f"current_progress : {start} / {end}")
                    substr = word_str[start:end]
                    if substr in self.vocab:
                        # 첫 토큰이 아니고 원래 단어의 일부인 경우 '##' 추가
                        if start > 0:
                            tokens.append("##" + substr)
                        else:
                            tokens.append(substr)
                        start = end
                        found = True
                        success_count += 1
                        break
                    end -= 1

                # 매칭되는 것이 없으면 한 글자씩 처리
                if not found:
                    fail_count += 1
                    if start > 0:
                        tokens.append("##" + word_str[start])
                    else:
                        tokens.append(word_str[start])
                    start += 1

            result.append(tokens)
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            for tokens in result:
                f.write(" ".join(tokens) + "\n")

        end_time = time.time()
        self.summary["average token length"] = sum(len(tokens) for tokens in result) / len(result) if result else 0
        self.summary["토큰화 시간"] = end_time - start_time
        self.summary["vocab"] = self.vocab_file_path
        self.summary["대상 파일 명"] = self.infer_file_path
        self.summary["최대 반복 횟수"] = self.max_iteration
        self.summary["최소 등장 횟수"] = self.min_pair_freq
        self.summary["성공 횟수"] = success_count
        self.summary["실패 횟수"] = fail_count

        def save_summary():
            try:
                with open("total_output.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []

            # 새로운 데이터 추가
            data.append(self.summary)

            # 전체 데이터 저장
            with open("total_output.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        save_summary()


if __name__ == "__main__":

    tokenizer = Tokenizer()
    tokenizer.max_iteration = 10000
    tokenizer.min_pair_freq = 40
    tokenizer.vocab_file_path = f"vocab_{0}_{0}_special_chars_false.txt"
    tokenizer.output_file_path = f"output_{0}_{0}_special_chars_false.txt"

    asyncio.run(tokenizer.update_vocab())
    tokenizer.tokenize()


async def main():
    tokenizer = Tokenizer()

    asyncio.run(tokenizer.update_vocab(remove_special_chars=False))
    tokenizer.tokenize()


if __name__ == "__main__":
    asyncio.run(main())
