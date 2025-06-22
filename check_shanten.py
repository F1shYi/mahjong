from multiprocessing import Pool, cpu_count
import random
import copy
import json
from MahjongGB import MahjongShanten


def generate_single_tilewall(args):
    """
    生成单个牌墙的函数，用于多进程
    修改为从牌墙末尾开始分配手牌
    """
    target_shanten, num_players, ori_tileWall = args

    hands = [[] for _ in range(num_players)]
    cur_player = 0
    tileWall = ori_tileWall.copy()
    attempt = 0

    while attempt < 1000000:
        attempt += 1
        random.shuffle(tileWall)

        # 从牌墙末尾开始分配手牌
        cur_hand = tileWall[-13:]  # 取最后13张牌
        if MahjongShanten(pack=tuple(), hand=tuple(cur_hand)) == target_shanten:
            hands[cur_player] = cur_hand
            tileWall = tileWall[:-13]  # 移除最后13张牌
            cur_player += 1

        if cur_player == num_players:
            # 记录手牌
            # 因为从牌墙末尾开始分配手牌，所以需要反转手牌方便查看
            check_hands = [" ".join(reversed(hd)) for hd in hands]

            # 分配剩余牌
            tiles_per_player = len(tileWall) // num_players
            for i in range(num_players):
                start_idx = i * tiles_per_player
                end_idx = start_idx + tiles_per_player
                extracted_tiles = tileWall[start_idx:end_idx]
                extracted_tiles.extend(hands[i])
                hands[i] = extracted_tiles

            final_tilewall = []
            for i in range(num_players):
                final_tilewall.extend(hands[i])

            return {
                'tilewall': ' '.join(final_tilewall),
                'hands': check_hands
            }

    return None


def target_shanten_tilewall(target_shanten=1, num_players=4, num_tilewall=100):
    """
    多进程版本的牌墙生成函数
    """

    # 生成原始牌墙
    ori_tileWall = []
    for j in range(4):
        for i in range(1, 10):
            ori_tileWall.append('W' + str(i))
            ori_tileWall.append('B' + str(i))
            ori_tileWall.append('T' + str(i))
        for i in range(1, 5):
            ori_tileWall.append('F' + str(i))
        for i in range(1, 4):
            ori_tileWall.append('J' + str(i))

    # 准备多进程参数
    num_processes = min(cpu_count(), 4)  # 限制最大进程数
    collections = []

    # 每5个牌墙为一组进行处理
    for batch_start in range(0, num_tilewall, 5):
        batch_size = min(5, num_tilewall - batch_start)
        args = [(target_shanten, num_players, ori_tileWall)
                for _ in range(batch_size)]

        # 使用进程池并行处理
        with Pool(processes=num_processes) as pool:
            results = pool.map(generate_single_tilewall, args)

        # 过滤掉None结果（生成失败的）
        batch_collections = [r for r in results if r is not None]
        collections.extend(batch_collections)

        # 存储当前批次的数据
        with open(f'curriculum_data/shanten_tilewalls_{target_shanten}.json', 'w', encoding='utf-8') as f:
            json.dump({target_shanten: collections},
                      f, indent=4, ensure_ascii=False)

        # 清理内存
        del results
        del batch_collections

    return collections


if __name__ == '__main__':
    import os
    os.makedirs('curriculum_data', exist_ok=True)
    for ts in range(1, 6):
        collections = target_shanten_tilewall(
            target_shanten=ts, num_players=4, num_tilewall=5000)
