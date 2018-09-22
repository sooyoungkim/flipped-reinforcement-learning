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
from gridWorld.valueIteration.environment import GraphicDisplay, Env


class ValueIteration:
    """
    따로 정책 발전이 필요없다. 시작부터 최적 정책을 가정하기때문이다.
    """
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
        # 감가율
        self.discount_factor = 0.9

    def value_iteration(self):
        # 새로운 가치함수 초기화
        next_value_table = [[0.0] * self.env.width for _ in range(self.env.height)]

        """
        벨만 최적 방정식
        ----------------------
        결과 : 최적 가치함수

            (1) 가치함수를 최적 정책에 대한 가치 함수라고 가정하고 (시작부터 최적 정책을 가정하므로 따로 policy improvement 가 필요없다.)
            (2) 반복적으로 계산하면
            (3) 결국 최적 정책에 대한 참 가치함수, 즉 '최적 가치함수'를 찾게 되는 것이다.
        """
        # '모든 상태(좌표)'에 대해 '벨만 최적 방정식'을 계산
        for state in self.env.get_all_states():
            # 마침 상태(좌표)이면 가치 함수 값 = 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue

            value_list = []  # 가치 함수를 위한 빈 리스트

            # 현재 에이전트는 '상,하,좌,우'로 행동이 가능
            # 가능한 모든 행동에 대해서 큐함수 계산
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                # 큐함수 : '보상 + (감가율 * 상태변환확률(여기서는 1로 가정) * 다음 상태 가치함수)' 을 계산
                value_list.append((reward + self.discount_factor * next_value))

            # 특정 상태에 대해 최댓값을 다음 가치 함수로 대입 (소수점 둘째 자리까지 표현)
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)

        # '모든 상태'에 대해 벨만 최적 방정식의 계산이 끝나면 현재 가치함수를 업데이트해준다. => 새로운 가치함수를 얻게된다.
        self.value_table = next_value_table

    """
    특정 상태(state)에서 정책(policy)에 따른 행동(action) 반환
       - 화면에서 'MOVE 버튼'을 눌러 최적 정책 계산 과정을 진행하도록 되어있다. 하나 하나 직접 확인해보며 실행할 수 있다.
       - 현재 정책에 따라서 움직이기 위해 사용된다.
    """
    def get_action(self, state):
        # 마침 상태(좌표)이면 빈 리스트 반환
        if state == [2, 2]:
            return []

        max_value = -99999
        max_action_list = []

        """
        명시적으로(explicit) 정책(policy)이 표현되는 것이 아니고 (즉, 정책이 독립적으로 존재하지 않는다. get_policy 메소드도 없다.)
        가치함수 안에 정책(policy)이 내재돼(implicit) 있다.
        
        """
        # 현재 에이전트는 '상,하,좌,우'로 행동이 가능
        # (벨만 최적 방정식을 통해 구한 가치함수를 토대로) 가능한 모든 행동에 대해서 큐함수 값을 계산해서 비교 -> 탐욕 정책 사용
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            # 큐함수 = '보상 + (감가율 * 상태변환확률(여기서는 1로 가정) * 다음 상태 가치함수)' 을 계산
            value = (reward + self.discount_factor * self.get_value(next_state))

            # 받을 보상이 최대인 행동만(복수일 경우 여러 개)을 추려낸다.
            if value > max_value:
                max_value = value
                max_action_list.clear()
                max_action_list.append(action)
            elif value == max_value:
                max_action_list.append(action)

        return max_action_list

    """
    상태에 따른 가치 함수 값 반환
    """
    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)


if __name__ == "__main__":
    # 환경에 대한 객체 선언
    env = Env()
    # 에이전트에게는 환경에 대한 정보가 필요하다.
    value_iteration = ValueIteration(env)

    # 사용자가 주는 입력에 따라 에이전트가 역할을 수행한다.
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()