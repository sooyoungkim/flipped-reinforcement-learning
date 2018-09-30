##############################################################################################################
# Policy 를 인공신경망으로 근사해보자.
#
# "장애물이 움직이는 문제"
# - 에이전트 목표가 움직이는 장애물을 피해서 도착지점에 가는 것이라면?
#   - 장애물은 3개로 기존보다 하나 늘어나고 같은 속도와 방향으로 움직인다.
#       - 속도가 같다는 것은 한 스텝마다 한 칸씩 움직인다는 것이고,
#       - 장애물들은 왼쪽이나 오른쪽 벽에 부딪힐 경우에 다시 튕겨져 나와서 반대 방향으로 움직인다.
#   - 에이전트가 장애물을 만날 경우 보상은 (-1)이며 도착지점에 도달할 경우 보상은 (+1)이다.
#
# 가치함수를 토대로 행동을 선택하고 가치함수를 업데이트 하면서 학습하는 방법이 아니라
#   상태에 따라 바로 행동을 선택하는 Policy-based Reinforcement Learning, 즉 Policy 를 직접적으로 근사시킨다.
##############################################################################################################
import copy
import numpy as np
from gridWorld.policyGradient.environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 1    # 2500


# 그리드월드 예제에서의 REINFORCE 에이전트
class Reinforce:
    def __init__(self):
        # 이미 학습 완료된 모델로 게임 플레이할지 여부
        self.load_model = False

        # 상태의 크기와 행동의 크기 정의
        # - 인공신경망 입력
        self.state_size = 15
        # - 인공신경망 출력 : 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.states, self.actions, self.rewards = [], [], []
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

        if self.load_model:
            # 학습 완료된 모델 불러오기
            self.model.load_weights('./save_model/reinforce_trained.h5')

    """
    '상태가 입력이고 (근사된)Policy(각 행동을 할 확률) 가 출력'인 인공신경망 생성

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 24)                384  <- (1+15) * 24
    _________________________________________________________________
    dense_2 (Dense)              (None, 24)                600  <- (1+24) * 24
    _________________________________________________________________
    dense_3 (Dense)              (None, 5)                 125  <- (1+24) *  5 
    =================================================================
    Total params: 1,109     <- 384 + 600 + 125
    Trainable params: 1,109
    Non-trainable params: 0
    _________________________________________________________________
    """
    def build_model(self):
        model = Sequential()
        # 은닉층 2개
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        # 출력층 1개
        # 활성화 함수로 softmax 함수를 사용한다.
        #   L softmax 함수 : 출력 값이 다 합해서 1이 된다
        #   L policy 정의가 각 행동을 할 확률이므로 이 확률들을 합하면 1이 된다.
        model.add(Dense(self.action_size, activation='softmax'))
        # 모델정보 출력
        model.summary()

        # 오차함수와 가중치 최적화는 optimizer 메소드에서 직접 구현

        return model

    """
    정책신경망을 업데이트 하기 위한 오류함수와 훈련함수 형태 구축
    """
    def build_optimizer(self):
        # placeholder tensor 생성
        action = K.placeholder(shape=[None, 5])             # Tensor("Placeholder:0", shape=(?, 5), dtype=float32)
        discounted_rewards = K.placeholder(shape=[None, ])  # Tensor("Placeholder_1:0", shape=(?,), dtype=float32)

        """
        (1) 크로스 엔트로피 오류(손실)함수 생성
            : 현재 예측값이 정답과 얼마나 가까운지를 나타낸다. 두 값이 가까워질수록 크로스 엔트로피 값은 줄어든다.
        """
        # '실제로 선택한 행동을 할 확률'을 정답으로 둔다.
        #    1) action : 5개의 행동 중에서 선택한 행동은 1, 나머지는 0인 리스트이다.
        #    2) self.model.output 은 정책신경망의 출력 값인 Policy 이다.
        action_prob = K.sum(action * self.model.output, axis=1)     # Tensor("Sum:0", shape=(?,), dtype=float32)
        cross_entropy = K.log(action_prob) * discounted_rewards     # Tensor("mul_1:0", shape=(?,), dtype=float32)
        loss = -K.sum(cross_entropy)                                # Tensor("Neg:0", shape=(), dtype=float32)

        """
        (2) 최적화 함수 생성 : 최적 업데이트 기울기값 계산
        """
        optimizer = Adam(lr=self.learning_rate)
        # updates = optimizer.get_updates(self.model.trainable_weights, [], loss)   # Old interface: (params, constraints, loss)
        updates = optimizer.get_updates(loss, self.model.trainable_weights)         # New interface: (loss, params)

        """
        (3) 정책신경망을 업데이트하는 훈련함수 생성
            Arguments
                1> inputs: List of placeholder tensors. 적용) [self.model.input, action, discounted_rewards]
                2> outputs: List of output tensors.     적용) []
                3> updates: List of update ops.         적용) updates=updates
        """
        train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)

        return train

    """
    정책신경망으로 행동 선택
        정책 자체가 확률적이기 때문에 그 확률에 따라 행동을 선택하면된다.
            L environment 에서 타임스텝마다 -0.1의 보상을 준다. -> 가만히 시작점에서 머무는 행동이 좋은 행동이 아닌 것임을 알 수 있게한다.
    """
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    """
    하나의 에피소드 동안 지나온 모든 상태에 대해 각각 반환값 계산
    """
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    """
    하나의 에피소드 동안의 상태, 행동, 보상을 저장
    """
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)    # 0으로 초기화
        act[action] = 1                     # 5개의 행동 중에서 '선택한 행동을 1'로 설정(나머지는 0)
        self.actions.append(act)

    """
    정책신경망 업데이트
    """
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        #
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    # 환경과 에이전트의 생성
    env = Env()
    agent = Reinforce()

    # 총 타임스텝 : 에이전트가 환경에서 얼마만큼 학습을 진행했는지 알 수 있다.
    total_step = 0

    for e in range(EPISODES):
        """
        환경 초기화 (상태 15개 정의)

        에이전트를 (0,0)으로 에이전트로부터 얼마나 떨어져 있는지에 대한 상대적인 위치 및 보상과 방향을 각각 상태로 정의한다.
        __________________________________________________________
        특징        상대적위치 x     상대적위치 y      보상      방향         
        ==========================================================
        장애물1        0             1            -1       -1
        __________________________________________________________
        장애물2        1             2            -1       -1
        __________________________________________________________
        장애물3        2             3            -1       -1
        __________________________________________________________
        도착점         4             4             1        X
        __________________________________________________________
        """
        state = env.reset()                 # [0, 1, -1, -1, 1, 2, -1, -1, 2, 3, -1, -1, 4, 4, 1]
        state = np.reshape(state, [1, 15])  # [[ 0  1 -1 -1  1  2 -1 -1  2  3 -1 -1  4  4  1]]

        done = False
        score = 0

        while not done:
            total_step += 1

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            # 환경으로 부터 다음 상태(next_state)와 보상(reward)을 받는다.
            #   L state)는 리스트, 보상(reward)은 숫자, 완료 여부(done)는 boolean
            next_state, reward, done = env.step(action)
            # 케라스 입력 형태 맞추기 : 형상을 (15,) -> (1, 15) 로 변형
            next_state = np.reshape(next_state, [1, 15])

            """
            '에피소드마다 실제로 얻은 보상(Return Gt)으로 학습'하는 Policy Gradient 이다.
                - Return 값을 사용하므로 Monte Carlo Policy Gradient 라고도 부른다. 
            """
            # sample <s,a,r> -> 에피소드가 끝날 때까지 기다리면 에피소드가 지나온 상태에 대해 각각의 반환값을 구할 수 있다.
            agent.append_sample(state, action, reward)

            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # 에피소드마다 정책신경망 업데이트
                agent.train_model()
                # 에피소드마다 학습 결과 출력
                print("episode:", (e + 1), "  score:", round(score, 2), "  time_step:", total_step)

        # # 100 에피소드마다 학습 결과 출력 및 모델 저장
        # if e % 100 == 0:
        #     agent.model.save_weights("./save_model/reinforce.h5")
