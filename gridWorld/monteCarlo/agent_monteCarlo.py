##########################################################################################################
# Monte Carlo Estimation
# - DP 에서는 기본적으로 environment dynamics 을 안다고 가정하였으나, 이제 이것을 가정하지 않는다.
#   sample 에 의한 experience 만 얻을 수 있는 정보이다.
# - 무작위(random) sample 을 기반으로 complete return (not partial return) 에 대한 평균(mean)을 구한다.
#       L MC는 bootstrap 이 아니다.
#               L bootstrap : 다른 state의 state-value 예상값에 의존하여 현재 state 의 state-value 값을 예측하는 방법
# - 더 많은 return 을 얻을수록 그 평균치는 기대치에 수렴할 것이다.
#
# (1) 경험을 통한 학습 :
#       MC method 는 environment 와의 모의(simulated) 또는 실제(actual) interaction 을 통해 얻은
#       states, action, rewards 가 포함된 sample 이라고 불리우는 경험(experience)을 이용한다.
# (2) 에피소드 단위의 학습 :
#       episodic tasks 에 대해 적용하는 것을 가정한다. 즉, step-by-step 보다는 episode-by-episode 단위의 학습이다.
#
# Monte Carlo Method
# (1) First-visit MC method :
#       하나의 episode 에서 state s를 처음 visit 후 얻은 return 값의 평균을 사용해서 Vㅠ(s)를 구한다.
#       (하나의 episode 에서 다시 접한 state s는 state-value function 평균치를 구하는데 사용하지 않는다.)
# (2) Every-visit MC method :
#       하나의 episode 에서 state s의 모든 visit 을 고려하여 return 값의 평균을 계산한다.
##########################################################################################################
import numpy as np
import random
from collections import defaultdict
from gridWorld.monteCarlo.environment import Env


class MonteCarlo:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions      # 상,하,좌,우 -> [0, 1, 2, 3]
        self.learning_rate = 0.01   # step size
        self.discount_factor = 0.9  # 감가율
        self.epsilon = 0.1          # 입실론
        self.sampling = []           # 하나의 episode 에 대한 모든 trajectory 의 next_state, reward, done 저장
        self.value_table = defaultdict(float)   # 에이전트가 방문한 state 의 value function 값 저장

    # 메모리에 sample 저장
    def save_sample(self, state, reward, done):
        self.sampling.append([state, reward, done])

    """
    에피소드에서 에이전트가 방문한 state 의 value function 업데이트
    
    <first visit method> 
            samples: [[[0, 0], 0, False],
                      [[0, 0], 0, False],
                      [[0, 0], 0, False],
                      [[1, 0], 0, False],
                      [[0, 0], 0, False],
                      [[1, 0], 0, False],
                      [[2, 0], 0, False],
                      [[2, 0], 0, False],
                      [[2, 0], 0, False],
                      [[2, 1], -100, True]]
            계산되는 sample: [[2, 1], -100, True]
                           [[2, 0], 0, False]
                           [[1, 0], 0, False]
                           [[0, 0], 0, False]
    """
    def update(self):
        return_s = 0
        visit_state = []

        # 하나의 에피소드의 모든 샘플에 대해
        print("samples : ", self.sampling)
        for sample in reversed(self.sampling):
            state_s = str(sample[0])    # 예) '[1,2]'

            # 첫번째 visit 에 대한 것만 평균값 계산 -> 두번 이상 방문한 state with Return G는 계산되지 않는다.
            if state_s not in visit_state:
                print("-> 계산되는 sample : ", sample)
                visit_state.append(state_s)
                # reverse 해서 사용하므로 G6 -> G5 -> G4 -> G3 -> G2 -> G1의 형태로 계산된다. (책 44P)
                return_s = self.discount_factor * (sample[1] + return_s)
                # 현재 state 의 value function 값
                value_s = self.value_table[state_s]
                # return 평균값으로 state 의 value function 값 업데이트
                self.value_table[state_s] = (value_s + self.learning_rate * (return_s - value_s))  # (PT 2P)

    """
    입실론 탐욕 정책에 따라서 action 반환
        :param      state  -> [x,y]
        :return     action -> 0 ~ 3 중 하나
    """
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 입실론 보다 적으면 랜덤 action([0, 1, 2, 3]) 중 하나 반환
            action = np.random.choice(self.actions)
        else:
            # state 에서 취할 수 있는 action 의 value function 값 얻기
            value = self.possible_state_value(state)
            # value function 이 최대값인 action 반환
            action = self.arg_max(value)
        return int(action)

    @staticmethod
    def arg_max(args):
        max_index_list = []
        max_value = args[0]   # 첫 번째 값을 max 설정해서 비교

        # 최대값 계산
        for index, value in enumerate(args):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        # 무작위로 인덱스 하나 반환
        return random.choice(max_index_list)

    """
    state 에서 취할 수 있는 action 의 value function 값 반환
    :param      state  -> [x,y]
    :return     state_value : 모든 action(상,하,좌,우)의 value function 값을 담은 리스트
    """
    def possible_state_value(self, state):
        col, row = state
        state_value = [0.0] * 4     # [0.0, 0.0, 0.0, 0.0]

        # [1] 상
        if row != 0:
            state_value[0] = self.value_table[str([col, row - 1])]
        else:
            state_value[0] = self.value_table[str(state)]   # 그리드 범위를 넘어가면 현재 state 적용

        # [2] 하
        if row != self.height - 1:
            state_value[1] = self.value_table[str([col, row + 1])]
        else:
            state_value[1] = self.value_table[str(state)]

        # [3] 좌
        if col != 0:
            state_value[2] = self.value_table[str([col - 1, row])]
        else:
            state_value[2] = self.value_table[str(state)]

        # [4] 우
        if col != self.width - 1:
            state_value[3] = self.value_table[str([col + 1, row])]
        else:
            state_value[3] = self.value_table[str(state)]

        # print("next_state = ", next_state)
        # [0.0, 0.0, 0.0, -0.7290000000000001]
        #  ...
        return state_value


if __name__ == "__main__":
    env = Env()
    agent = MonteCarlo(actions=list(range(env.n_actions)))  # 4 : 상(up), 하(down), 좌(left), 우(right)

    # generate an episodes
    for episode in range(10):   # 1000
        state = env.reset()

        # 입실론 탐욕 정책에 따라서 특정 state 의 action 얻기
        action = agent.get_action(state)

        while True:
            env.render()

            # 현재 state에서 action 취하기
            # 환경으로 부터 다음 상태(next_state)와 보상(reward)을 받는다.
            #   L state)는 리스트, 보상(reward)은 숫자, 완료 여부(done)는 boolean
            next_state, reward, done = env.step(action)
            # 메모리에 sample 저장 : 몬테카를로 예측에서는 Rt+1, St+1 을 샘플로 사용한다.
            agent.save_sample(next_state, reward, done)
            # 다음 상태(state)에서 취할 수 있는 action 얻기
            action = agent.get_action(next_state)

            # 하나의 에피소드가 완료되어야만 지나온 모든 state 의 value function 을 업데이트한다.
            # 에피소드가 끝이 없거나 길이가 긴 경우에는 몬테카를로 예측은 적합하지 않다.
            if done:
                agent.update()
                agent.sampling.clear()
                break

