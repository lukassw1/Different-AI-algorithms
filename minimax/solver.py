from Pick import Pick, PickState
from copy import deepcopy
from player import Player
from table import TABLE_H
from random import choice
import matplotlib.pyplot as plt
WIN = 7000


def function_h(current_state: PickState, player_max: Player):
    value = 0
    if current_state.get_winner() is not None:
        if current_state.get_winner() == player_max:
            return -WIN
        else:
            return WIN
    else:
        for number in current_state.current_player_numbers:
            value += TABLE_H[number]
        if current_state.get_current_player() == player_max:
            return -value
        return value


class Solver():
    """A solver. It may be initialized with some hyperparameters."""
    def __init__(self, heuristic_function) -> None:
        self.heuristic_function = heuristic_function
        pass

    def minimax(self, cur_state: Pick, depth: int, player_max: Player, alfa=float("-inf"), beta=float("inf")):
        if cur_state.is_finished() or depth == 0:
            return self.heuristic_function(cur_state.state, player_max), None
        successors = cur_state.get_moves()
        if cur_state.state.get_current_player() == player_max:
            value = float("-inf")
            for successor_move in successors:
                next_state = deepcopy(cur_state)
                next_state.make_move(successor_move)
                new_value, path = self.minimax(deepcopy(next_state), depth-1, Player(0), alfa, beta)
                if new_value > value:
                    best_move = successor_move
                elif new_value == value:
                    best_move = choice([successor_move, best_move])
                value = max(value, new_value)
                alfa = max(alfa, value)
                if value >= beta:
                    break
            return value, best_move
        else:
            value = float("inf")
            for successor_move in successors:
                next_state = deepcopy(cur_state)
                next_state.make_move(successor_move)
                new_value, path = self.minimax(deepcopy(next_state), depth-1, Player(1), alfa, beta)
                if new_value < value:
                    best_move = successor_move
                elif new_value == value:
                    best_move = choice([successor_move, best_move])
                value = min(value, new_value)
                beta = min(beta, value)
                if value <= alfa:
                    break
            return value, best_move

    def solve(self, player1_depth, player2_depth):
        game = Pick()
        while not game.is_finished():
            # player 1 turn
            val1, move1 = self.minimax(game, player1_depth, Player(1))
            game.make_move(move1)
            if move1 is None: break
            if game.is_finished(): break
            # player 2 turn
            val2, move2 = self.minimax(game, player2_depth, Player(0))
            if move2 is None: break
            game.make_move(move2)
        return game.get_winner()


def random_maxmin_ind(values, maxmin):
        if maxmin:
            target = max(values)
        else:
            target = min(values)
        inds = []
        for i in range(len(values)):
            if values[i] == target:
                inds.append(i)
        ind = choice(inds)
        return ind


def explore_sample(solver1: Solver, sample, player1depth, player2depth):
    results = []
    for i in range(sample):
        r = solver1.solve(player1depth, player2depth)
        results.append(r)
    return results


def show_result(result):
    print(f"Player 1 wins - {result.count(Player(1))}\nPlayer 2 wins - {result.count(Player(0))}\nDraws - {result.count(None)}")


def analyse_sample(solver1, sample, p1d, p2d):
    results = explore_sample(solver1, sample, p1d, p2d)
    show_result(results)
    plot_result(results, sample, p1d, p2d)


def plot_result(result, sample, player1depth, player2depth):
    labels = 'Player 1', 'Player 2', 'Draw'
    data = [result.count(Player(1)), result.count(Player(0)), result.count(None)]
    explode = (0, 0.1, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',  
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title(f"Pick34 winners for the data:\n[sample={sample}, player 1 depth={player1depth}, player 2={player2depth}")
    plt.savefig(f"plot{sample}_{player1depth}_{player2depth}.png")


def main():
    my_solver = Solver(function_h)
    analyse_sample(my_solver, 100, 3, 4)


if __name__ == "__main__":
    main()
