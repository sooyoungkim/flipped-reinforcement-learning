##############################################################################################################
# Q-Learning (Off-Policy Temporal-Difference Control)
# - 살사는 On-Policy 즉, 자신이 행동하는 대로 학습한다.
#       L <s, a, r, s', a'> 일때
#           다음상태 s'에서 실제 선택한 행동이 초록색 세모(-100)로 가는 안 좋은 행동일때
#           그 정보가 현재상태 s의 큐함수를 업데이트 할때 포함된다.
#           이 이후에 다시 현재상태 s에 에이전트가 오게 되면 에이전트는 행동 a를 하는 것이 안 좋다고 판단한다.
#           (상태 s에서 행동 a의 큐함수가 값이 낮을 것이므로)
#
# - 탐험은 절대적으로 필요한 부분!
#
# - Off-Policy :
#       행동하는 정책(behavior policy)과 학습하는 정책(target policy)을 따로 분리한다.
#       에이전트는 behavior policy 로 지속적인 탐험을 하면서
#               target policy 로 학습한다. (독립적)
#                   L 다음상태 s'에서 실제 선택한 행동이 초록색 세모(-100)로 가는 안 좋은 행동이라도
#                       그 정보가 현재상태 s의 큐함수를 업데이트 할때 포함되지 않는다.
##############################################################################################################
import numpy as np
import random
from gridWorld.QLearning.environment import Env
from collections import defaultdict


class QLearning:
    def __init__(self, actions):
        # 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s'> 샘플로부터 큐함수 값을 업데이트
    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        # 벨만 최적 방정식을 사용한 큐함수 값의 업데이트
        # 현재상태 s의 큐함수를 업데이트할 때는 다음상태 s'의 최대 큐함수값을 이용한다.
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    """
    입실론 탐욕 정책에 따라서 action 반환
        :param      state  -> [x,y]
        :return     action -> 0 ~ 3 중 하나
    """
    def get_action(self, state):
        if np.random.rand() > self.epsilon:
            # 입실론 보다 적으면 무작위 행동 반환 - action([0, 1, 2, 3]) 중 하나 랜덤 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수 값이 최대인 action 반환
            value = self.q_table[state]
            action = self.arg_max(value)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]

        # 최대값 계산
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        # 무작위로 인덱스 하나 반환
        return random.choice(max_index_list)


if __name__ == "__main__":
    env = Env()
    agent = QLearning(actions=list(range(env.n_actions)))

    # generate an episodes
    for episode in range(10):   # 1000
        # 게임 환경 초기화
        state = env.reset()

        while True:
            env.render()

            # 입실론 탐욕 정책에 따라서 특정 state 의 action 얻기
            action = agent.get_action(str(state))
            # 현재 state에서 action 취하기
            # 환경으로 부터 다음 상태(next_state)와 보상(reward)을 받는다.
            #   L state)는 리스트, 보상(reward)은 숫자, 완료 여부(done)는 boolean
            next_state, reward, done = env.step(action)

            # sample <s,a,r,s'> 로 큐함수 값 업데이트 -> 매 타임스템마다 큐함수 값을 업데이트 한다.
            # 에이전트가 다음상태 s'를 일단 알게되면 학습한다.
            #   L 실제로 다음상태 s'에서 어떤 행동을 했는지와 상관없이 다음상태 s'의 최대 큐함수값을 이용해서 학습한다.
            agent.learn(str(state), action, reward, str(next_state))

            state = next_state

            # 모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break
