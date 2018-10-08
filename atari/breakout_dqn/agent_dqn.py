from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import gym

EPISODES = 5    # 50000

"""
학습속도를 높이기 위해 흑백화면으로 전처리
   브레이크아웃의 화면은 색상을 포함하므로 게임화면 이미지는 210 X 160 X 3 이다.
          - 에이전트가 학습할 떄는 색상을 알 필요는 없으므로 흑백으로 바꾼다.
          - 게임화면에서 불필요한 부분들을 잘라내서 이미지 크기를 줄인다.
   
    :param  observe : 전처리 전의 이미지 = 하나의 게임화면
    :return processed_observe : 전처리 거친 (84 X 84 X 1) 이미지 

"""
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


# 브레이크아웃에서의 DQN 에이전트
class DQN:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)   # 히스토리 길이는 4 프레임
        self.action_size = action_size
        # DQN 하이퍼파라미터 : 1 부터 0.1 까지 백만 스텝 동안 시간이 지남에 따라 감소시킨다.
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.discount_factor = 0.99
        # 미니 배치 크기
        self.batch_size = 32
        # 5만 스텝 이후부터 학습 시작
        self.train_start = 50000
        # 일만 스텝에 한번씩 타겟모델 업데이트
        self.update_target_rate = 10000
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        # 게임초반 정지상태 상한선
        self.no_op_steps = 30

        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        # 두 모델의 갸중치를 통일
        self.update_target_model()

        self.optimizer = self.optimizer()
        self.episode_q_max, self.episode_loss = 0, 0

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        # self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        # self.summary_writer = tf.summary.FileWriter('summary', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn_trained.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # (오류함수로 MSE 를 사용하는 것이 아니라) Huber Loss 를 이용하기 위해 최적화 함수를 직접 정의
    # 후버로스 : -1~1 사이의 구간에서는 2차함수이며 그 밖의 구간에서는 1차함수인 오류함수.
    #          -1 과 1 을 넘어가는 큰 오류에 대해 민감하게 반응하지 않아도 된다는 장점이 있어서 더 안정적으로 학습이 진행된다.
    def optimizer(self):
        # 플레이스홀더 생성
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        # 가중치 업데이트 최적화 함수로 RMSprop 사용
        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        # < 함수형태 생성 >
        #   입력 : [신경망입력값, 행동, 정답]
        #   출력 : [손실(오류값)]
        #   각 가중치 업데이트 값 : RMSprop 으로 각 가중치 업데이트 값 구하기
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        # 원래 픽셀 값인 0~255 사이의 값이 아니라 0~1 사이의 값으로 정규화
        history = np.float32(history / 255.0)

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'> 을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        # 학습마다 입실론 감소
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # 리플레이 메모리에서 무작위로 '배치크기'의 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        # 미리 형태를 지정
        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))   # (5,) 형상으로 target 생성
        action, reward, dead = [], [], []

        # 지정해둔 형태에 무작위로 추출한 샘플을 차례로 저장
        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)        # 정규화
            next_history[i] = np.float32(mini_batch[i][3] / 255.)   # 정규화
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        # 신경망을 통해 예측값을 가져오기
        target_value = self.target_model.predict(next_history)

        # 배치크기만큼 정답(target)을 생성
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                # 정답 계산
                target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        # 모델 업데이트 : 최적화 함수를 호출 - 함수의 입력값은 [현재 상태 히스토리, 행동, 정답]이고 출력값은 [손실(오류값)]이다.
        loss = self.optimizer([history, action, target])
        self.episode_loss += loss[0]


if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQN(action_size=3)  # 정지, 왼쪽, 오른쪽

    scores, episodes, total_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        episode_step, episode_score, episode_life = 0, 0, 5
        observe = env.reset()

        """
         브레이크아웃을 플레이해보면 -> 공이 날아올떄 일정한 방향으로만 날아온다는 것. (보통 왼쪽 아니면 오른쪽 구석)
                                 따라서 에이전트가 가장 처음에 빠지는 오류는 구석에 붙는 것이다.
                                   -> 이를 방지하기 위해 초반에 일정 기간 동안 아무것도 하지 않는 구간을 무작위로 설정,
                                      그 상한선은 no_op_steps 에 세팅
        """
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)  # action=1은 정지를 의미, 즉 아무것도 하지 않는 행동으로 환경에서 진행

        # 각 에피소드마다 시작 상태 전처리
        state = pre_processing(observe)
        # 히스토리 길이는 4 프레임
        # 하지만 처음 시작할 때는 연속된 4 프레임의 화면을 모을 수 없으므로 하나의 이미지만으로 히스토리를 만들게 된다.
        # 전처리 후 state 는 (84 X 84 X 1) 의 형태이므로 axis=2 로 채널(1)에 해당하는 축에 대해 state 를 쌓아줘야한다.
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            total_step += 1
            episode_step += 1

            # 바로 전 4개의 상태(히스토리)로 행동을 선택
            action = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            #   : 전처리전의 다음 상태, 보상, 에피소드가 끝났는지 여부, 현재 에피소드에서 몇 개의 목숨이 남았는지 정보(1개의 에피소드마다 5번)
            observe, reward, done, info = env.step(real_action)

            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            # 전처리를 거친 새로운 상태(next_state)와 오래된 상태를 버린 히스토리를 axis=3에 대해 더한다.
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # 신경망으로 부터 히스토리에 대한 큐함수 예측값을 얻고 최대값을 구한다.
            agent.episode_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])

            # 목숨을 잃은 경우에 대한 처리
            if episode_life > info['ale.lives']:
                dead = True
                episode_life = info['ale.lives']

            # clip 함수 : 보상은 1을 넘을 수 있는데, 1을 넘는 보상을 전부 1로 만들어주는 함수이다.
            #            게임마다 점수를 주는 기준이 다르므로 다른 아타리 게임에도 이 DQN 을 적용해보기 위함(생략해도 학습에 문제 없다)
            reward = np.clip(reward, -1., 1.)

            # 샘플 <s, a, r, s'> 을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)
            if len(agent.memory) >= agent.train_start:
                # 학습은 첫 5만 스텝이 지난 후부터 진행하도록 설정
                agent.train_model()

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if total_step % agent.update_target_rate == 0:
                agent.update_target_model()

            episode_score += reward

            if dead:
                # 초기화
                dead = False
            else:
                # 계속 진행
                history = next_history

            if done:
                # # 각 에피소드 당 학습 정보를 기록
                # if total_step > agent.train_start:
                #     stats = [episode_score,
                #              agent.episode_q_max / float(episode_step),
                #              episode_step,
                #              agent.episode_loss / float(episode_step)]
                #     for i in range(len(stats)):
                #         agent.sess.run(agent.update_ops[i],
                #                        feed_dict={
                #                            agent.summary_placeholders[i]: float(stats[i])
                #                        })
                #     summary_str = agent.sess.run(agent.summary_op)
                #     agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  score:", episode_score,
                      "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon,
                      "  total_step:", total_step,
                      "  average_q:", agent.episode_q_max / float(episode_step),
                      "  average loss:", agent.episode_loss / float(episode_step))

                # 초기화
                agent.episode_q_max, agent.episode_loss = 0, 0

        # # 1000 에피소드마다 모델 저장
        # if e % 1000 == 0:
        #     agent.model.save_weights("./save_model/breakout_dqn.h5")
