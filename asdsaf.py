import asyncio
from typing import List


async def process_chunks_and_get_pairs(train_data: List[List[str]], best_pair: tuple = None, new_vocab: str = None) -> tuple:
    chunk_size = len(train_data) // 10
    divided_train_data = [train_data[i : i + chunk_size] for i in range(0, len(train_data), chunk_size)]

    async def process_chunk(chunk: List[List[str]]):
        pairs = {}
        updated_chunk = []

        for word in chunk:
            updated_word = word.copy()
            if best_pair:  # 업데이트 모드
                for i in range(len(word) - 1):
                    if tuple(word[i : i + 2]) == best_pair:
                        updated_word[i : i + 2] = [new_vocab]

            # pairs 계산
            for i in range(len(updated_word) - 1):
                pair = tuple(updated_word[i : i + 2])
                pairs[pair] = pairs.get(pair, 0) + 1

            updated_chunk.append(updated_word)

        return pairs, updated_chunk

    async with asyncio.taskgroups.TaskGroup() as tg:
        task_list = [tg.create_task(process_chunk(chunk)) for chunk in divided_train_data]
    results = [chunk.result() for chunk in task_list]

    # 결과 병합
    all_pairs = {}
    updated_data = []
    for pairs, chunk in results:
        for pair, count in pairs.items():
            all_pairs[pair] = all_pairs.get(pair, 0) + count
        updated_data.extend(chunk)

    return all_pairs, updated_data
