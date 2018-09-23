######################################################################################
# 다이내믹 프로그래밍
# - 순차적 행동 결정 문제(MDP)를 벨만 방정식을 통해 푼다.
#       L 벨만 기대 방정식을 이용한 것은 policy iteration이고
#       L 벨만 최적 방정식을 이용한 것이 value iteration
# - 반복적으로(iterative) 가치함수를 구한다.
#       L V0(s) -> V1(s) -> V2(s) ... -> Vk(s) -> ... -> Vㅠ(s)
#
# 벨만 방정식 :
#   - '현재 상태의 가치함수'와 '다음 상태의 가치함수' 사이의 관계식이다.
#   - 벨만 기대 방정식 : 특정 정책을 따라갔을때 가치함수 사이의 관계식
#   - 벨만 최적 방정식 : 최적 정책을 따라갔을때 가치함수 사이의 관계식
######################################################################################
######################################################################################
# 다이내믹 프로그래밍의 한계
#    - 계산을 빠르게 하는 것이지 "학습"을 하는 것은 아니다. 즉, 머신러닝이 아니다.
#
# < 한계 3가지>
# (1) 계산 복잡도 : state 크기의 3제곱에 비례한다.
#                   예) 그리드월드에서는 5 X 5
# (2) 차원의 저주(curse of dimensionality)
#              : state 의 차원이 늘어나면 state 수가 지수적으로 증가할 것이다.
#                   예) 그리드월드에서는 (x,y)로 표현되는 2차원
# (3) 환경에 대한 완벽한 정보가 필요
#       - 환경의 모델(입력과 출력 사이의 방정식)에 해당하는 '보상'과 '상태변환 확률'을 정확히 안다는 가정
#           -> 보통은 이 정보를 정확히 알 수 없다는 문제!
#
# 모델없이 (model free) 환경과의 상호작용을 통해 입력과 출력 사이의 관계를 학습하는 방법
# => '환경을 모르지만' '환경과의 상호작용을 통해 경험을 바탕으로' '학습하는' 방법
# ==> 강화학습이 등장한다.
######################################################################################
# -*- coding: utf-8 -*-
import random
from gridWorld.policyIteration.environment import GraphicDisplay, Env


class PolicyIteration:
    def __init__(self, env):
        # 환경 객체 생성
        self.env = env

        # 2차원 리스트로 가치함수 초기화
        #   L 가치함수 : 현재의 정책을 따라갔을 때 받을 보상에 대한 기대값
        #   L 그리드월드 가로 x 그리드월드 세로 크기로 생성 : DP 에서는 에이전트가 '모든 상태'에 대해 가치함수를 계산한다.
        #       (1) [0.0] * 5 = [0.0, 0.0, 0.0, 0.0, 0.0]
        #       (2) [0.0] * 5 for _ in range(5) = [[0.0, 0.0, 0.0, 0.0, 0.0],
        #                                          [0.0, 0.0, 0.0, 0.0, 0.0],
        #                                          [0.0, 0.0, 0.0, 0.0, 0.0],
        #                                          [0.0, 0.0, 0.0, 0.0, 0.0],
        #                                          [0.0, 0.0, 0.0, 0.0, 0.0]]
        self.value_table = [[0.0] * env.width for _ in range(env.height)]

        # 그리드월드의 각 state(좌표)마다 '상, 하, 좌, 우'에 해당하는 action 에 대해 동일한 확률(25%)로 정책 초기화
        # (1) [[0.25, 0.25, 0.25, 0.25]] * 5 = [[0.25, 0.25, 0.25, 0.25],
        #                                       [0.25, 0.25, 0.25, 0.25],
        #                                       [0.25, 0.25, 0.25, 0.25],
        #                                       [0.25, 0.25, 0.25, 0.25],
        #                                       [0.25, 0.25, 0.25, 0.25]]
        # (2) [[0.25, 0.25, 0.25, 0.25]] * 5 for _ in range(5) = [[
        #                                                           [0.25, 0.25, 0.25, 0.25],
        #                                                           [0.25, 0.25, 0.25, 0.25],
        #                                                           [0.25, 0.25, 0.25, 0.25],
        #                                                           [0.25, 0.25, 0.25, 0.25],
        #                                                           [0.25, 0.25, 0.25, 0.25]
        #                                                         ],
        #                                                         ... ,
        #                                                         [
        #                                                           [0.25, 0.25, 0.25, 0.25],
        #                                                           [0.25, 0.25, 0.25, 0.25],
        #                                                           [0.25, 0.25, 0.25, 0.25],
        #                                                           [0.25, 0.25, 0.25, 0.25],
        #                                                           [0.25, 0.25, 0.25, 0.25]
        #                                                         ]]
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        self.policy_table[2][2] = []    # 마침 상태의 설정

        # 감가율
        self.discount_factor = 0.9

    """
    정책을 어떻게 평가할까요?
        - 가치함수가 정책이 얼마나 좋은지 판단하는 근거가 된다.
        - 가치함수 : 현재의 정책을 따라갔을때 받을 보상에 대한 기댓값, 정책의 가치 
    """
    def policy_evaluation(self):
        # 새로운 가치함수 초기화
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]

        """
        벨만 기대 방정식
        ----------------------
        결과 : 현재 정책을 따라 갔을때 받을 참 보상(TRUE Reward)

            (1) 가치함수를 현재 정책에 대한 가치함수라고 가정하고
            (2) 반복적으로 계산하면
            (3) 결국 현재 정책에 대한 참 가치함수가 된다.
        """
        # '모든 상태(좌표)'에 대해 '벨만 기대 방정식'을 계산
        for state in self.env.get_all_states():
            value = 0.0
            # 마침 상태(좌표)이면 가치 함수 값 = 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            # 현재 에이전트는 '상,하,좌,우'로 행동이 가능
            # 가능한 모든 행동에 대해서 '벨만 기대 방정식' 계산
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                # 벨만 기대 방정식 : 'ㅠ(a|s) * state-value function' 을 계산해서 더하는 것을 반복한다. => 기대값을 계산한 것이 된다.
                value += self.get_policy(state)[action] * (
                        reward + self.discount_factor * self.get_value(next_state))

            next_value_table[state[0]][state[1]] = round(value, 2)

        # '모든 상태'에 대해 벨만 기대 방정식의 계산이 끝나면 현재 가치함수를 업데이트해준다. => 새로운 가치함수를 얻게된다.
        self.value_table = next_value_table

    """
    정책평가를 바탕으로 어떻게 정책을 발전시킬 수 있을까요?
        - 현재 가치 함수에 대해서 탐욕 정책 발전(Greedy Policy Improvement)
        - 탐욕 정책 발전 : 가치가 가장 높은 하나의 행동을 선택하는 것이다.
                        현재 상태에서 가장 좋은 행동이 여러 개일 수도 있다 -> 가장 좋은 행동들을 동일한 확률로 선택(무작위 선택)

    """
    def policy_improvement(self):
        # 새로운 정책 초기화
        next_policy = self.policy_table

        # '모든 상태(좌표)'에 대해
        for state in self.env.get_all_states():
            # 마침 상태(좌표)이면 pass
            if state == [2, 2]:
                continue

            max_value = -99999
            max_index_list = []
            result = [0.0, 0.0, 0.0, 0.0]

            # 현재 에이전트는 '상,하,좌,우'로 행동이 가능
            # 가능한 모든 행동에 대해서 '탐욕 정책' 발전 사용
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                # state-value function : '보상 + (감가율 * 상태변환확률(여기서는 1로 가정) * 다음 상태 가치함수)' 을 계산
                value = reward + self.discount_factor * self.get_value(next_state)

                # 받을 보상이 최대인 행동의 index 만을 추려낸다.
                if value == max_value:
                    max_index_list.append(index)
                elif value > max_value:
                    max_value = value
                    max_index_list.clear()
                    max_index_list.append(index)

            # 특정 상태에 대해서 최대 보상을 받게하는 행동의 확률을 계산
            #   L 받을 보상이 최대인 행동의 index 가 여러 개라면 해당하는 행동을 동일한 확률로 선택(무작위 선택)하게 한다.
            prob = 1 / len(max_index_list)
            for index in max_index_list:
                result[index] = prob

            # 특정 상태에 대한 정책 업데이트
            next_policy[state[0]][state[1]] = result

        # '모든 상태'에 대해 탐욕 정책 발전 적용이 끝나면 현재 정책을 업데이트해준다. => 새로운 정책을 얻게된다.
        self.policy_table = next_policy

    """
    특정 상태(state)에서 정책(policy)에 따른 행동(action) 반환
       - 화면에서 'MOVE 버튼'을 눌러 최적 정책 계산 과정을 진행하도록 되어있다. 하나 하나 직접 확인해보며 실행할 수 있다.
       - 현재 정책에 따라서 움직이기 위해 사용된다.
    """
    def get_action(self, state):
        # 0 ~ 1 사이의 값을 무작위로 추출
        random_pick = random.randrange(100) / 100

        # 특정 상태에 대한 정책 가져오기
        policy = self.get_policy(state)

        # todo : 특정 상태에 대해 확률이 가장 큰 행동을 우선 추려내고 그 중에 무작위 선택을 해야하는데...
        # 정책에 담긴 행동('상,하,좌,우') 중에 무작위로 하나의 행동을 추출
        policy_sum = 0.0
        for index, value in enumerate(policy):
            # 정책의 확률을 차례대로 더해간다
            policy_sum += value
            # 무작위로 추출한 수를 넘게하는 행동의 인덱스를 반환한다
            if random_pick < policy_sum:
                return index

    """
    상태에 따른 정책 반환
    ----------------------
        :return 특정 상태에서 '상, 하, 좌, 우' action 에 대한 확률
    """
    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        # ㅠ(a|s) : 특정 상태에서 특정 행동을 할 확률
        return self.policy_table[state[0]][state[1]]

    """
    상태에 따른 가치 함수 값 반환
    """
    def get_value(self, state):
        # 소숫점 둘째 자리까지만 계산
        return round(self.value_table[state[0]][state[1]], 2)


"""
dynamic programming 에서 에이전트는 "환경의 모든 정보를 알고 있다".
    - 이 정보를 통해 에이전트는 최적 정책을 찾는 계산을 하는 것
"""
if __name__ == "__main__":
    # 환경에 대한 객체 선언
    env = Env()
    # 에이전트에게는 환경에 대한 정보가 필요하다.
    agent = PolicyIteration(env)

    # 사용자가 주는 입력에 따라 에이전트가 역할을 수행한다.
    grid_world = GraphicDisplay(agent)
    grid_world.mainloop()
