##############################################################################################################
# SARSA (On-Policy Temporal-Difference Control)
#
# (1) 경험을 통한 학습 :
#       <state, action, reward, next state, next action> 가 포함된 sample 이라고 불리우는 경험(experience)을 이용한다.
# (2) 타임스텝 단위(step-by-step)의 학습 :
#       실시간 예측 : 에피소드가 끝날 때까지 기다릴 필요 없이 하나의 sample 이 생성되면 그 sample 로 바로 큐함슈를 업데이트 한다.
#
# Value Iteration 과 같이 별도의 정책을 두지 않고 에이전트는 현재 상태에서 가장 큰 가치를 지니는 행동을 선택하는 탐욕 정책을 사용한다.
# - model-free 이면 '큐함수'를 구하는 것이 유용하다.
#       L 따라서 SARSA 는 bootstrap 이다.
#           L bootstrap : 다른 state 의 value 예상값에 의존하여 현재 state 의 value 값을 예측하는 방법
#       L 문제는 한번도 접하지 않은 state-action pair 가 있을 수 있다는 것이다.
#           L 끊임없는 exploration 을 보장해야 한다. => 탐욕 정책의 대안인 '입실론 탐욕 정책'을 따른다.
##############################################################################################################
import numpy as np
import random
from collections import defaultdict
from gridWorld.sarsa.environment import Env


class SARSA:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s', a'> 샘플로부터 큐함수 값을 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate * (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q

    """
    입실론 탐욕 정책에 따라서 action 반환
        :param      state  -> [x,y]
        :return     action -> 0 ~ 3 중 하나
    """
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 입실론 보다 적으면 무작위 행동 반환 - action([0, 1, 2, 3]) 중 하나 랜덤 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수값이 최대인 action 반환
            value = self.q_table[state]
            action = self.arg_max(value)
        return action

    @staticmethod
    def arg_max(action_value):
        max_index_list = []
        max_value = action_value[0]

        # 최대값 계산
        for index, value in enumerate(action_value):
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
    agent = SARSA(actions=list(range(env.n_actions)))

    # generate an episodes
    for episode in range(10):   # 1000
        # 게임 환경 초기화
        state = env.reset()

        # 입실론 탐욕 정책에 따라서 특정 state 의 action 얻기
        action = agent.get_action(str(state))

        while True:
            env.render()

            # 현재 state에서 action 취하기
            # 환경으로 부터 다음 상태(next_state)와 보상(reward)을 받는다.
            #   L state)는 리스트, 보상(reward)은 숫자, 완료 여부(done)는 boolean
            next_state, reward, done = env.step(action)
            # 다음 상태(state)에서 취할 수 있는 action 얻기
            next_action = agent.get_action(str(next_state))

            # sample <s,a,r,s',a'> 로 큐함수 값 업데이트 -> 매 타임스템마다 큐함수 값을 업데이트 한다.
            agent.learn(str(state), action, reward, str(next_state), next_action)

            state = next_state
            action = next_action

            # 모든 큐함수 값을 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break
