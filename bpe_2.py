import asyncio
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

        self.vocab = self.get_vocab()
        self.summary = ""

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

    async def update_chunks(self, train_data: List[List[str]], best_pair: tuple, new_vocab: str) -> List[List[str]]:
        # 비동기 처리를 통해 속도의 향상
        chunk_size = len(train_data) // 10
        divided_train_data = [train_data[i : i + chunk_size] for i in range(0, len(train_data), chunk_size)]

        async def process_chunk(chunk: List[List[str]]):
            for word in chunk:
                for i in range(len(word) - 1):
                    if tuple(word[i : i + 2]) == best_pair:
                        word[i : i + 2] = [new_vocab]
            return chunk

        async with asyncio.taskgroups.TaskGroup() as tg:
            task_list = [tg.create_task(process_chunk(chunk)) for chunk in divided_train_data]
        results = [chunk.result() for chunk in task_list]
        # 각 chunk의 결과를 하나의 리스트로 결합
        return [item for sublist in results for item in sublist]

    async def update_vocab(self):
        time_start = time.time()

        # 학습데이터를 기반으로 사전을 업데이트
        train_data = self.get_txt_data(self.train_file_path)
        train_data: List[List[str]] = self.preprocess(train_data)

        # 중간 병합 과정
        iteration = 0
        min_pair_freq = MIN_PAIR_FREQ  # 최소 빈도수 임계값
        max_iterations = ASYNC_MAX_ITERATION  # 최대 반복횟수

        print("max_iterations : ", max_iterations)
        while True:
            pairs = self.get_pairs(train_data)
            if not pairs:
                stop_reason = "pairs is empty"
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_pair_freq:  # 빈도수가 너무 낮으면 종료
                stop_reason = "stop by low frequency"
                break

            new_vocab = "".join(best_pair)
            train_data = await self.update_chunks(train_data, best_pair, new_vocab)

            iteration += 1
            curr_time = time.time()
            print(f"iteration : {iteration} / {max_iterations}, time : {curr_time - time_start}")

            if iteration >= max_iterations:
                stop_reason = "stop by max iteration"
                break

        final_pairs = self.get_pairs(train_data)

        # 빈도수 기준으로 정렬하여 상위 max_vocab개 선택
        sorted_pairs = sorted(final_pairs.items(), key=lambda x: x[1], reverse=True)
        vocab = ["".join(pair) for pair, _ in sorted_pairs[: self.max_vocab]]

        time_end = time.time()
        print("time : ", time_end - time_start)

        self.summary = {
            "학습 총 시간": time_end - time_start,
            "최대 반복 횟수": max_iterations,
            "실제 반복횟수": iteration,
            "stop_reason": stop_reason,
            "max vocab": self.max_vocab,
            "vocab 크기": len(vocab),
        }
        print(self.summary)
        self.save_vocab(vocab)

    def tokenize(self):
        # 입력받은 텍스트 데이터를 vocab를 기반으로 토큰화
        start_time = time.time()
        input_data = self.get_txt_data(self.infer_file_path)
        input_data: List[str] = self.preprocess(input_data)

        # vocab 기반으로 토큰화

        result = []
        for word in input_data:
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
                        break
                    end -= 1

                # 매칭되는 것이 없으면 한 글자씩 처리
                if not found:
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
        self.summary["대상 파일 명"] = self.infer_file_path
        print(self.summary)

    def get_pairs(self, text_list: List[List[str]]) -> dict:
        # 연속된 문자 쌍의 빈도수를 계산
        pairs = {}
        for word in text_list:
            for i in range(len(word) - 1):
                # 리스트 대신 튜플 사용
                pair = tuple(word[i : i + 2])  # 리스트를 튜플로 변환
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs


tokenizer = Tokenizer()
asyncio.run(tokenizer.update_vocab())

tokenizer.tokenize()
