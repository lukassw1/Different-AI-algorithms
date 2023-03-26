from itertools import combinations


AIM = 34
N = 4
LAST_NUM = N**2

def make_table_h() -> dict:
    win_possibilities = [possibile for possibile in combinations(range(1, LAST_NUM+1), N) if sum(possibile) == AIM]
    win_number_amount = []
    for possibile in win_possibilities:
        win_number_amount += possibile
    table = {x: win_number_amount.count(x) for x in range(1, LAST_NUM+1)}
    return table


TABLE_H = {
    1: 19,
    2: 20,
    3: 21,
    4: 22,
    5: 22,
    6: 23,
    7: 23,
    8: 22,
    9: 22,
    10: 23,
    11: 23,
    12: 22,
    13: 22,
    14: 21,
    15: 20,
    16: 19
}
