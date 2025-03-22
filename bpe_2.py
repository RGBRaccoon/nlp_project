import asyncio
import json
import re
import time
from typing import List

from default_setting import *


class Tokenizer:
    def __init__(self):
        self.max_vocab = MAX_VOCAB

        self.vocab_file_path = VOCAB_FILE_PATH
        self.train_file_path = TRAIN_FILE_PATH
        self.infer_file_path = INFER_FILE_PATH
        self.output_file_path = OUTPUT_FILE_PATH
        self.min_pair_freq = MIN_PAIR_FREQ
        self.max_iteration = ASYNC_MAX_ITERATION

        self.remove_special_chars = True
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

    def preprocess(self, text: str) -> List[List[str]]:
        if self.remove_special_chars:
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

    async def process_chunks_and_get_pairs(self, train_data: List[List[str]], best_pair: tuple = None, new_vocab: str = None) -> tuple:
        chunk_size = len(train_data) // self.worker_num
        divided_train_data = [train_data[i : i + chunk_size] for i in range(0, len(train_data), chunk_size)]

        async def process_chunk(chunk: List[List[str]]):
            pairs = {}
            updated_chunk = []

            for word in chunk:
                if best_pair:  # 업데이트 모드
                    for i in range(len(word) - 1):
                        if tuple(word[i : i + 2]) == best_pair:
                            word[i : i + 2] = [new_vocab]

                # pairs 계산
                for i in range(len(word) - 1):
                    pair = tuple(word[i : i + 2])
                    pairs[pair] = pairs.get(pair, 0) + 1

                updated_chunk.append(word)

            return pairs, updated_chunk

        async with asyncio.taskgroups.TaskGroup() as tg:
            task_list = [tg.create_task(process_chunk(chunk)) for chunk in divided_train_data]
        results = [chunk.result() for chunk in task_list]

        all_pairs = {}
        updated_data = []
        for pairs, chunk in results:
            for pair, count in pairs.items():
                all_pairs[pair] = all_pairs.get(pair, 0) + count
            updated_data.extend(chunk)

        return all_pairs, updated_data

    async def update_vocab(self):
        time_start = time.time()

        # 학습데이터를 기반으로 사전을 업데이트
        train_data = self.get_txt_data(self.train_file_path)
        train_data: List[List[str]] = self.preprocess(train_data)
        iteration = 0

        while True:
            print(f"iteration : {iteration}/{self.max_iteration}")
            pairs, train_data = await self.process_chunks_and_get_pairs(train_data)
            if not pairs:
                stop_reason = "pairs is empty"
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_pair_freq:  # 빈도수가 너무 낮으면 종료
                stop_reason = "stop by low frequency"
                break

            new_vocab = "".join(best_pair)
            pairs, train_data = await self.process_chunks_and_get_pairs(train_data, best_pair, new_vocab)

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
        self.remove_special_chars = False
        input_data: List[str] = self.preprocess(input_data)

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


async def run_tokenizer(max_iter: int, min_freq: int, vocab_path: str, output_path: str):
    tokenizer = Tokenizer()  # 각 작업마다 새로운 객체 생성
    tokenizer.max_iteration = max_iter
    tokenizer.min_pair_freq = min_freq
    tokenizer.vocab_file_path = vocab_path
    tokenizer.output_file_path = output_path
    await tokenizer.update_vocab()
    tokenizer.tokenize()


async def main():
    # chunk_size = 4

    # for i in range(10):
    #     for chunk_start in range(0, 4, chunk_size):
    #         async with asyncio.taskgroups.TaskGroup() as tg:
    #             for j in range(chunk_start, min(chunk_start + chunk_size, 4)):
    #                 max_iter = 10000 + i * 1000
    #                 min_freq = MIN_PAIR_FREQ - 10 * j
    #                 vocab_path = f"vocab_{i}_{j}.txt"
    #                 output_path = f"output_{i}_{j}.txt"

    #                 tg.create_task(run_tokenizer(max_iter, min_freq, vocab_path, output_path))

    for i in range(0, 2):
        for j in range(0, 4):
            tokenizer = Tokenizer()
            tokenizer.max_iteration = 10000 + i * 1000
            tokenizer.min_pair_freq = MIN_PAIR_FREQ - 10 * j
            tokenizer.vocab_file_path = f"vocab_{i}_{j}.txt"
            tokenizer.output_file_path = f"output_{i}_{j}.txt"
            # await run_tokenizer(max_iter, min_freq, vocab_path, output_path)
            tokenizer.tokenize()


if __name__ == "__main__":
    asyncio.run(main())
