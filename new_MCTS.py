import numpy as np
import random
import sys
import utils
from five_stone_game import main_process as five_stone_game
import time

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

distrib_calculater = utils.distribution_calculater(utils.board_size)


class edge:
    def __init__(self, action, parent_node, priorP):
        """
        self.action_value 指的是backpropagation的胜率。如果为0.0，说明没有backpropagation过
        :param action:
        :param parent_node:
        :param priorP:
        """
        self.action = action    # 该 edge 的落子
        self.counter = 1.0      # 该 edge 被探索过的次数
        self.parent_node = parent_node
        self.priorP = priorP    # 神经网络计算出的先验概率
        self.child_node = None  # self.search_and_get_child_node()
        self.action_value = 0.0 # 探索过的胜利的次数

    def backup(self, v):  # back propagation
        self.action_value += v
        self.counter += 1
        self.parent_node.backup(-v)

    def get_child(self):
        """
        当该 edge的child_node为None时，说明需要继续往下探索（expand），所以返回True
        """
        if self.child_node is None:
            # 如果该 edge 还没有子节点，则生成其对应的子节点
            self.counter += 1
            self.child_node = node(self, -self.parent_node.node_player)
            return self.child_node, True
        else:
            self.counter += 1
            return self.child_node, False

    def UCB_value(self):  # 计算当前的UCB value
        # self.action_value 为0.0，说明还没有backpropagation过
        if self.action_value:
            Q = self.action_value / self.counter
        else:
            Q = 0
        return Q + utils.Cpuct * self.priorP * np.sqrt(self.parent_node.counter) / (1 + self.counter)


class node:
    def __init__(self, parent, player):
        self.parent = parent
        self.counter = 0.0
        self.child = {}
        self.node_player = player

    def add_child(self, action, priorP):  # 增加node治下的一个edge，但是没有实际创建新的node
        action_name = utils.move_to_str(action)
        self.child[action_name] = edge(action=action, parent_node=self, priorP=priorP)

    def get_child(self, action):
        child_node, _ = self.child[action].get_child()
        return child_node

    def eval_or_not(self):
        return len(self.child) == 0

    def backup(self, v):  # back propagation
        self.counter += 1
        if self.parent:
            self.parent.backup(v)

    def get_distribution(self, train=True):  ## used to get the step distribution of current
        for key in self.child.keys():
            distrib_calculater.push(key, self.child[key].counter)
        return distrib_calculater.get(train=train)

    def UCB_sim(self):  # 用于根据UCB公式选取node
        UCB_max = -sys.maxsize
        UCB_max_key = None
        for key in self.child.keys():
            if self.child[key].UCB_value() > UCB_max:
                UCB_max_key = key
                UCB_max = self.child[key].UCB_value()
        # 注意 self.child[key] 是edge，因此需要调用 get_child() 获取实际的子节点
        this_node, expand = self.child[UCB_max_key].get_child()
        return this_node, expand, self.child[UCB_max_key].action


class MCTS:
    def __init__(self, board_size=11, simulation_per_step=400, neural_network=None):
        self.board_size = board_size
        self.s_per_step = simulation_per_step
        # self.database = {0: {"":node(init_node, 1, self)}}  # here we haven't complete a whole database that can be
        # self.current_node = self.database[0][""]                   # used to search the exist node
        self.current_node = node(None, 1)
        self.NN = neural_network
        self.game_process = five_stone_game(board_size=board_size)  # 这里附加主游戏进程
        self.simulate_game = five_stone_game(board_size=board_size)  # 这里附加用于模拟的游戏进程

        self.distribution_calculater = utils.distribution_calculater(self.board_size)

    def renew(self):
        self.current_node = node(None, 1)
        self.game_process.renew()

    def MCTS_step(self, action):
        next_node = self.current_node.get_child(action)
        next_node.parent = None
        return next_node

    def simulation(self):  # simulation的程序
        eval_counter, step_per_simulate = 0, 0
        # 通过 self.s_per_step控制探测未知节点的深度
        for _ in range(self.s_per_step):
            expand, game_continue = False, True
            this_node = self.current_node
            self.simulate_game.simulate_reset(self.game_process.current_board_state(True))
            state = self.simulate_game.current_board_state()
            # selection阶段
            # 之所以用循环是要一直探测到未被探测过的节点为止(expand为true)，这样就进入expansion阶段
            while game_continue and not expand:
                # 给this_node补子edge。如果已被检测过(有子edge）,则跳过
                if this_node.eval_or_not():
                    # 返回所有点（即使该点已经下了）的概率。返回的state_prob是1✖64的张量
                    state_prob, _ = self.NN.eval(
                        utils.transfer_to_input(state, self.simulate_game.which_player(), self.board_size))
                    # 根据现在棋局state，返回有效落子点
                    valid_move = utils.valid_move(state)
                    eval_counter += 1
                    # 根据有效落子点，筛选state_prob，给this_node增加子edge
                    for move in valid_move:
                        this_node.add_child(action=move, priorP=state_prob[0, move[0] * self.board_size + move[1]])

                # 根据UCB公式向下选择一层：计算下一步的action、更新this_node、以及该node是否有子node（expand）
                # 返回的expand决定是继续selection还是expansion
                this_node, expand, action = this_node.UCB_sim()
                # 模拟计算下一步的action执行后，游戏是否结束
                game_continue, state = self.simulate_game.step(action)
                step_per_simulate += 1

            if not game_continue:
                # 如果游戏停止，直接 backup阶段
                this_node.backup(1)
            elif expand:
                # expansion阶段。（这里并没有探测到终局，只探测一层就返回）
                _, state_v = self.NN.eval(
                    utils.transfer_to_input(state, self.simulate_game.which_player(), self.board_size))
                # backup阶段
                this_node.backup(state_v)
        return eval_counter / self.s_per_step, step_per_simulate / self.s_per_step

    def game(self, train=True):  # 主程序
        game_continue = True
        game_record = []
        begin_time = int(time.time())
        step = 1
        total_eval = 0
        total_step = 0
        while game_continue:
            begin_time1 = int(time.time())
            avg_eval, avg_s_per_step = self.simulation()
            action, distribution = self.current_node.get_distribution(train=train)
            game_continue, state = self.game_process.step(utils.str_to_move(action))
            self.current_node = self.MCTS_step(action)
            game_record.append({"distribution": distribution, "action": action})
            end_time1 = int(time.time())
            print("step:{},cost:{}s, total time:{}:{} Avg eval:{}, Aver step:{}".format(step, end_time1 - begin_time1,
                                                                                        int((
                                                                                                    end_time1 - begin_time) / 60),
                                                                                        (end_time1 - begin_time) % 60,
                                                                                        avg_eval, avg_s_per_step),
                  end="\r")
            total_eval += avg_eval
            total_step += avg_s_per_step
            step += 1
        self.renew()
        end_time = int(time.time())
        min = int((end_time - begin_time) / 60)
        second = (end_time - begin_time) % 60
        print("In last game, we cost {}:{}".format(min, second), end="\n")
        return game_record, total_eval / step, total_step / step

    def interact_game_init(self):
        self.renew()
        _, _ = self.simulation()
        action, distribution = self.current_node.get_distribution(train=False)
        game_continue, state = self.game_process.step(utils.str_to_move(action))
        self.current_node = self.MCTS_step(action)
        return state, game_continue

    def interact_game1(self, action):
        game_continue, state = self.game_process.step(action)
        return state, game_continue

    def interact_game2(self, action, game_continue, state):
        self.current_node = self.MCTS_step(utils.move_to_str(action))
        if not game_continue:
            pass
        else:
            # 模拟落子比较耗时
            _, _ = self.simulation()
            # 然后就是计算机选择落子
            action, distribution = self.current_node.get_distribution(train=False)
            game_continue, state = self.game_process.step(utils.str_to_move(action))
            self.current_node = self.MCTS_step(action)
        return state, game_continue

