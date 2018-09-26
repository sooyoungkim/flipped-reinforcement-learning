##############################################################################################################
# 살사 알고리즘 이용 & 큐함수를 인공신경망으로 근사해보자.
#
# "장애물이 움직이는 문제"
# - 에이전트 목표가 움직이는 장애물을 피해서 도착지점에 가는 것이라면?
#   - 장애물은 3개로 기존보다 하나 늘어나고 같은 속도와 방향으로 움직인다.
#       - 속도가 같다는 것은 한 스텝마다 한 칸씩 움직인다는 것이고,
#       - 장애물들은 왼쪽이나 오른쪽 벽에 부딪힐 경우에 다시 튕겨져 나와서 반대 방향으로 움직인다.
#   - 에이전트가 장애물을 만날 경우 보상은 (-1)이며 도착지점에 도달할 경우 보상은 (+1)이다.
##############################################################################################################
import copy
import random
import numpy as np
from gridWorld.deepSarsa.environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 2    # 555


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSA:
    def __init__(self):
        # 이미 학습 완료된 모델로 게임 플레이할지 여부
        self.load_model = False

        # 상태의 크기와 행동의 크기 정의
        # - 인공신경망 입력
        self.state_size = 15
        # - 인공신경망 출력 : 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)

        self.learning_rate = 0.001
        self.discount_factor = 0.99
        # 탐험(exploration) 보장
        self.epsilon = 1.
        # 입실론을 더 빨리 감소 시키면 점수는 더 빨리 수렴할 수 있지만 에이전트는 탐험을 덜 하게되므로 최적으로 수렴하지 않을 수 있다.
        self.epsilon_decay = .9999
        # 지속적인 탐험을 위해 0으로 만들지 않고 하한선만 정해둔다.
        self.epsilon_min = 0.01

        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.05
            # 학습 완료된 모델 불러오기
            self.model.load_weights('./save_model/deep_sarsa_trained.h5')

    """
    '상태가 입력이고 (근사된)큐함수가 출력'인 인공신경망 생성
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 30)                480  <- (1+15) * 30
    _________________________________________________________________
    dense_2 (Dense)              (None, 30)                930  <- (1+30) * 30
    _________________________________________________________________
    dense_3 (Dense)              (None, 5)                 155  <- (1+30) *  5  
    =================================================================
    Total params: 1,565
    Trainable params: 1,565
    Non-trainable params: 0
    _________________________________________________________________
    """
    def build_model(self):
        model = Sequential()
        # 은닉층 2개
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        # 출력층 1개
        model.add(Dense(self.action_size, activation='linear'))
        # 모델정보 출력
        model.summary()
        # 오차함수와 최적
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    """
    입실론 탐욕 정책에 따라서 action 선택
    -----------------------------------
    :param  state 에이전트를 (0,0)으로 에이전트로부터 장애물1,2,3과 도착점이 얼마나 떨어져 있는지 상대적 위치 및 보상과 방향 정보를 담은 상태
                    예) [[-4 -2 -1  1 -3 -1 -1 -1 -2  0 -1 -1  0  1  1]]
    """
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return random.randrange(self.action_size)
        else:
            # 모델로부터 행동 산출
            state = np.float32(state)
            q_values = self.model.predict(state)
            # 형태가 [[ 0.4208471  -0.58310926  0.00578364  0.09069238 -1.1442541 ]] 와 같으므로 0번째 값에서 최대값을 반환한다.
            return np.argmax(q_values[0])

    """
    큐함수를 근사하고 있는 인공신경망을 '경사하강법'을 사용해 업데이트 = 학습
    - 오차함수를 정의해야 한다. : 기본적으로 MSE 사용
    - 강화학습은 지도학습이 아니므로 정답이 있지 않다. 
        L 살사(SARSA)의 큐함수 업데이트 식에서 정답(target)의 역할과 예측의 역할을 나눠볼 수 있다.
            L 이 정답과 예측을 MSE 식에 대입해서 오차함수를 만들어 볼 수 있다.
    """
    def train_model(self, state, action, reward, next_state, next_action, done):
        # 시간이 지남에 따라 입실론 값을 감소시킨다. -> 500 에피소드 정도되면 입실론이 0.01이 되고 점수가 수렴한다.
        #   L 초반에는 에이전트가 탐험을 통해 다양한 상황에 대해 학습한다.
        #   L 학습이 충분히 이루어진 후에는 예측하는대로 에이전트가 움직인다.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 케라스의 입력은 float 형이여야한다.
        state = np.float32(state)
        next_state = np.float32(next_state)

        # 현재상태에 대한 (근사된) 큐함값
        # 형태가 [[ 0.4208471  -0.58310926  0.00578364  0.09069238 -1.1442541 ]] 와 같으므로 0번째 값을 가져온다.
        target = self.model.predict(state)[0]

        # <살사의 큐함수 업데이트 식 참고>
        if done:
            # 에피소드가 끝났으므로 즉각적인 보상만 고려한다.
            target[action] = reward
        else:
            # 정답의 역할 계산
            target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

        # 실제로 행동한 부분의 큐함수값만 바뀐다. -> 따라서 오류함수 계산에 실제로 행동된 큐함수값만 사용된다.
        # 예)
        #  before target: [-0.9403626   0.28477466 - 0.3009627   0.05266716 - 0.00641688]
        #  after  target: [ 0.25298813  0.28477466 - 0.3009627   0.05266716 - 0.00641688]

        # 케라스 사용하기위한 형상으로 변형
        target = np.reshape(target, [1, 5])
        # 인공신경망 업데이트 : 입력데이터(state)와 정답데이터(target)
        self.model.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    agent = DeepSARSA()

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

        # 입실론 탐욕 정책에 따라서 현재 상태에 대한 행동 선택
        action = agent.get_action(state)

        while not done:
            total_step += 1

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            # 환경으로 부터 다음 상태(next_state)와 보상(reward)을 받는다.
            #   L state)는 리스트, 보상(reward)은 숫자, 완료 여부(done)는 boolean
            next_state, reward, done = env.step(action)
            # 케라스 입력 형태 맞추기 : 형상을 (15,) -> (1, 15) 로 변형
            next_state = np.reshape(next_state, [1, 15])
            # 다음 상태(state)에서 취할 수 있는 action 선택
            next_action = agent.get_action(next_state)

            # sample <s,a,r,s',a'> 로 큐함수 값 업데이트 -> 매 타임스템마다 큐함수 값을 업데이트 한다.
            agent.train_model(state, action, reward, next_state, next_action, done)

            score += reward
            state = copy.deepcopy(next_state)
            action = next_action

            if done:
                # 에피소드마다 학습 결과 출력
                # <몇 번째 에피소드인지, 해당 에피소드에서 점수는 몇 점인지, 총 타임스텝, 입실론> 출력
                print("episode:", (e + 1), "  score:", score, "total_step", total_step, "  epsilon:", agent.epsilon)

        # # 100 에피소드마다 모델 저장
        # if e % 100 == 0:
        #     agent.model.save_weights("./save_model/deep_sarsa.h5")
