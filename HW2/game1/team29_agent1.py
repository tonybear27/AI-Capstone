
import STcpClient

import numpy as np
import random

class GameState:
    def __init__(self, board_size=12):
        self.board_size = board_size

    def map_flip(self, state):
        playerID, mapStat, sheepStat = state
        newMapStat = mapStat.copy()
        newSheepStat = sheepStat.copy()
        for i in range(len(mapStat)):
            for j in range(len(mapStat[0])):
                newMapStat[j][i] = mapStat[i][j]
                newSheepStat[j][i] = sheepStat[i][j]
        return (playerID, newMapStat, newSheepStat)
    
    def whether_onboard(self, x, y, board_size):
        return (x >= 0) and (x < board_size) and (y >= 0) and (y < board_size)


    def init(self, mapStat, board_size=12):

        def check(mapStat, init_pos):
            x, y = init_pos

            if mapStat[x][y] != 0:
                return False

            extended_map = np.pad(mapStat.copy(), pad_width=1, mode='constant', constant_values=0)
            window = extended_map[x:x + 3, y:y + 3]
            return np.any(window == -1)
        


        surroundings = [
                (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                (0, -2),  (0, -1),           (0, 1),  (0, 2),
                (1, -2),  (1, -1),  (1, 0),  (1, 1),  (1, 2),
                (2, -2),  (2, -1),  (2, 0),  (2, 1),  (2, 2)
            ]    
        final_init_pos = [0, 0]
        max_count = -1


        for i in range(board_size):
            for j in range(board_size):
                init_pos = [i, j]

                if not check(mapStat, init_pos):
                    continue

                count = 0
                for dx, dy in surroundings:
                    nx, ny = init_pos[0] + dx, init_pos[1] + dy
                    if self.whether_onboard(nx, ny, board_size) and mapStat[nx][ny] == 0:
                        count += 1

                if count > max_count:
                    max_count = count
                    final_init_pos = init_pos

        return final_init_pos


    def dirMapping(self, dir):
            table = { 1: (-1, -1), 2: (-1, 0), 3: (-1, 1), 4: (0, -1), 6: (0, 1), 7: (1, -1), 8: (1, 0), 9: (1, 1) }
            
            return table[dir]
    


    # add randomness to the game.
    def flip(self, action):
        table = {1: 1, 2: 4, 3: 7, 4: 2, 6: 8, 7: 3, 8: 6, 9: 9}
        x, y = action[0]
        m = action[1]
        dir = action[2]
        newx, newy = x, y
        newdir = table[dir]

        return [(newx, newy), m, newdir]

    
    # we have 9 options to take when doing every action
    def nextDir(self):
        return [1, 2, 3, 4, 6, 7, 8, 9]

    def opposite_xy(self, cur_pos):
        x, y = cur_pos
        return (y, x)



    def move(self, x, y, dir, mapStat):
        dx, dy = self.dirMapping(dir)
        while True:
            if not (0 <= x + dx < len(mapStat) and 0 <= y + dy < len(mapStat[0])):
                break
            if mapStat[x + dx][y + dy] != 0:
                break
            x += dx
            y += dy
        return (x, y)


    def act(self, state, action):
        playerID, mapStat, sheepStat = state
        newMapStat = mapStat.copy()
        newSheepStat = sheepStat.copy()

        x, y = action[0]
        m = action[1]
        dir = action[2]

        newx, newy = self.move(x, y, dir, mapStat)
        newMapStat[newx][newy] = playerID
        newSheepStat[x][y] -= m
        newSheepStat[newx][newy] = m

        newPlayerID = playerID % 4 + 1

        return (newPlayerID, newMapStat, newSheepStat)


    # get all the possible moves to consider 
    def possible_moves(self, state):
        playerID, mapStat, sheepStat = state
        moves = []
        height, width = len(mapStat), len(mapStat[0])
        splitable_sheep = []
        for i in range(height):
            for j in range(width):
                if mapStat[i][j] == playerID and int(sheepStat[i][j]) > 1:
                    splitable_sheep.append((i, j))
        for i, j in splitable_sheep:
            for dir in self.nextDir():
                newx, newy = self.move(i, j, dir, mapStat)
                if newx == i and newy == j:
                    continue
                m = int(sheepStat[i][j]) // 2
                moves.append([(i, j), m, dir])

        return moves

    # to check whether the node is a leaf node or not
    def isLeaf(self, state):
        return not self.possible_moves(state)

    # check whether the game is over or not 
    def isTerminal(self, state):
        for i in range(1, 5):
            if self.possible_moves((i, state[1], state[2])):
                return False
        return True

    def getTerritory(self, mapStat):
        territory = {1: 0, 2: 0, 3: 0, 4: 0}
        for i in range(len(mapStat)):
            for j in range(len(mapStat[0])):
                if 1 <= mapStat[i][j] <= 4:
                    territory[mapStat[i][j]] += 1
        return territory

    def dfs_search(self, mapStat, playerID, visited, i, j):
        if i < 0 or i >= self.board_size or j < 0 or j >= self.board_size or visited[i][j] or mapStat[i][j] != playerID:
            return 0    # invalid
        visited[i][j] = True
        return 1 + self.dfs_search(mapStat, playerID, visited, i - 1, j) \
            + self.dfs_search(mapStat, playerID, visited, i + 1, j) \
            + self.dfs_search(mapStat, playerID, visited, i, j - 1) \
            + self.dfs_search(mapStat, playerID, visited, i, j + 1)

    # get connected region to calculate the point
    def getAdjacentRegions(self, mapStat, playerID):
        adjacent_regions = []
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        for i in range(self.board_size):
            for j in range(self.board_size):
                if mapStat[i][j] == playerID and not visited[i][j]:
                    adjacent_regions.append(self.dfs_search(mapStat, playerID, visited, i, j))
        return adjacent_regions

    # calculate the 
    def getScore(self, state):
        playerID, mapStat, sheepStat = state
        regions = self.getAdjacentRegions(mapStat, playerID)
        return round(sum([region ** 1.25 for region in regions]))

    # return the team with the highest score in the game
    def getWinTeam(self, state):
            scores = [self.getScore((i, state[1], state[2])) for i in range(1, 5)]
            return 1 if scores[0] + scores[2] > scores[1] + scores[3] else 2

    # return the winner of the game 
    def getWinner(self, state):
        playerID, mapStat, sheepStat = state
        scores = [self.getScore((i, mapStat, sheepStat)) for i in range(1, 5)]
        return scores.index(max(scores)) + 1

  


###### classical MCTS implementation ###### 

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action 
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


class Tree:
    def __init__(self, state):
        self.root = Node(state)


class MCTS:
    def __init__(self, state, gameID):
        self.tree = Tree(state)
        self.playerID = state[0]
        self.gameID = gameID
        self.iterations = 100 if gameID == 2 else 300
        self.gamestate = GameState(15 if self.gameID == 2 else 12)

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda n: self.ucb(n))
        return node

    def expand(self, node):
        actions = self.possible_moves(node.state)
        for action in actions:
            new_state = self.act(node.state, action)
            new_node = Node(new_state, action, node)
            node.children.append(new_node)

    def simulate(self, state):
        while not self.gamestate.isTerminal(state):
            actions = self.gamestate.possible_moves(state)
            if not actions:
                state = (state[0] % 4 + 1, state[1], state[2])
                continue
            action = random.choice(actions)
            state = self.gamestate.act(state, action)

        if self.gameID == 4:
            player_team = 1 if self.playerID in [1, 3] else 2
            winning_team = self.gamestate.getWinTeam(state)
            return 1 if player_team == winning_team else -1
        else:
            winner = self.gamestate.getWinner(state)
            return 1 if winner == self.playerID else -1

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def ucb(self, node):
        if node.visits == 0:
            return float('inf')
        return node.value / node.visits + (2 * np.log(node.parent.visits) / node.visits) ** 0.5

    def get_action(self):
        root = self.tree.root
        for _ in range(self.iterations):
            selected_node = self.select(root)
            if not selected_node.children:
                self.expand(selected_node)

            expanded_node = selected_node if self.gamestate.isLeaf(selected_node.state) else random.choice(
                selected_node.children)
            value = self.simulate(expanded_node.state)
            self.backpropagate(expanded_node, value)

        best_child = max(root.children, key=lambda n: n.visits)  
        return best_child.action

    def possible_moves(self, state):
        return self.gamestate.possible_moves(state)

    def act(self, state, action):
        return self.gamestate.act(state, action)


def InitPos(mapStat):
    init_pos = GameState().init(mapStat)
    init_pos = GameState().opposite_xy(init_pos)
    return init_pos


def GetStep(playerID, mapStat, sheepStat):
    mcts = MCTS((playerID, mapStat, sheepStat), 1)
    action = mcts.get_action()
    action = GameState().flip(action)
    return action




############################# Server Processing #####################################
# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
