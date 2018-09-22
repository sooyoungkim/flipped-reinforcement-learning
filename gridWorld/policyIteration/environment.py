###############################################################################
#
# 그리드월드 문제의도
#   에이전트가 (2,1), (1,2) 위치를 피해 (2,2)에 도착해서 +1의 보상을 받는 것이다.
#   단순히 +1을 받는 것이 아니라 (2,2)에 도착하는 최적 정책을 찾는 것이 목표!
#
# 그리드월드의 경우 환경을 직접 만든 것이기때문에 environment.py 파일이 있다.
#   - 보통 강화학습을 적용할 때는 이미 구축되어 있는 환경에 에이전트를 생성해서 학습시키기 떄문에
#     에이전트에 관한 파일만 있어도 된다.
#   - 한 번쯤은 환경이 어떻게 구성되어 있는지 살표보는 것도 좋다.
#     강화학습을 적용하고 싶은 대상이 환경을 모두 구성해 놓지 않았을 수도 있기 때문이다.
#
###############################################################################
import tkinter as tk
from tkinter import Button
import time
import numpy as np
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100      # 픽셀 수
HEIGHT = 5      # 그리드월드 세로
WIDTH = 5       # 그리드월드 가로
TRANSITION_PROB = 1                 # 상태변환 확률을 무조건 1로 설정
POSSIBLE_ACTIONS = [0, 1, 2, 3]     # 에이전트가 할 수 있는 모든 행동 인덱스 : 0=좌, 1=우, 2=하, 3=상
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 좌표로 표현한 행동 (좌,우,하,상)
REWARDS = []


class GraphicDisplay(tk.Tk):
    """

    GUI로 그리드월드 환경을 보여주는 클래스

    """

    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Policy Iteration')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT + 50))
        self.texts = []
        self.arrows = []
        self.env = Env()
        self.agent = agent
        self.evaluation_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.text_reward(2, 2, "R : 1.0")
        self.text_reward(1, 2, "R : -1.0")
        self.text_reward(2, 1, "R : -1.0")

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # 버튼 초기화
        iteration_button = Button(self, text="Evaluate", command=self.evaluate_policy)
        iteration_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.13, HEIGHT * UNIT + 10, window=iteration_button)

        policy_button = Button(self, text="Improve", command=self.improve_policy)
        policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.37, HEIGHT * UNIT + 10, window=policy_button)

        policy_button = Button(self, text="Move", command=self.move_by_policy)
        policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.62, HEIGHT * UNIT + 10, window=policy_button)

        policy_button = Button(self, text="Reset", command=self.reset)
        policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.87, HEIGHT * UNIT + 10, window=policy_button)

        # 그리드 생성
        for col in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, HEIGHT * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        canvas.create_image(250, 150, image=self.shapes[1])
        canvas.create_image(150, 250, image=self.shapes[1])
        canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()

        return canvas

    def load_images(self):
        up = PhotoImage(Image.open("../img/up.png").resize((13, 13)))
        right = PhotoImage(Image.open("../img/right.png").resize((13, 13)))
        left = PhotoImage(Image.open("../img/left.png").resize((13, 13)))
        down = PhotoImage(Image.open("../img/down.png").resize((13, 13)))
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65, 65)))
        return (up, down, left, right), (rectangle, triangle, circle)

    """
    버튼 reset
    """
    def reset(self):
        if self.is_moving == 0:
            self.evaluation_count = 0
            self.improvement_count = 0
            for i in self.texts:
                self.canvas.delete(i)

            for i in self.arrows:
                self.canvas.delete(i)
            self.agent.value_table = [[0.0] * WIDTH for _ in range(HEIGHT)]
            self.agent.policy_table = ([[[0.25, 0.25, 0.25, 0.25]] * WIDTH
                                        for _ in range(HEIGHT)])
            self.agent.policy_table[2][2] = []
            x, y = self.canvas.coords(self.rectangle)
            self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

    def text_value(self, row, col, contents, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        origin_x, origin_y = 85, 70
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def text_reward(self, row, col, contents, font='Helvetica', size=10,
                    style='normal', anchor="nw"):
        origin_x, origin_y = 5, 5
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def rectangle_move(self, action):
        base_action = np.array([0, 0])
        location = self.find_rectangle()
        self.render()
        if action == 0 and location[0] > 0:  # 상
            base_action[1] -= UNIT
        elif action == 1 and location[0] < HEIGHT - 1:  # 하
            base_action[1] += UNIT
        elif action == 2 and location[1] > 0:  # 좌
            base_action[0] -= UNIT
        elif action == 3 and location[1] < WIDTH - 1:  # 우
            base_action[0] += UNIT
        # move agent
        self.canvas.move(self.rectangle, base_action[0], base_action[1])

    def find_rectangle(self):
        temp = self.canvas.coords(self.rectangle)
        x = (temp[0] / 100) - 0.5
        y = (temp[1] / 100) - 0.5
        return int(y), int(x)

    """
    버튼 move 
    """
    def move_by_policy(self):
        if self.improvement_count != 0 and self.is_moving != 1:
            self.is_moving = 1

            x, y = self.canvas.coords(self.rectangle)
            self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

            x, y = self.find_rectangle()
            while len(self.agent.policy_table[x][y]) != 0:
                self.after(100, self.rectangle_move(self.agent.get_action([x, y])))
                x, y = self.find_rectangle()
            self.is_moving = 0

    def draw_one_arrow(self, col, row, policy):
        if col == 2 and row == 2:
            return

        if policy[0] > 0:  # up
            origin_x, origin_y = 50 + (UNIT * row), 10 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.up))
        if policy[1] > 0:  # down
            origin_x, origin_y = 50 + (UNIT * row), 90 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.down))
        if policy[2] > 0:  # left
            origin_x, origin_y = 10 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.left))
        if policy[3] > 0:  # right
            origin_x, origin_y = 90 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.right))

    def draw_from_policy(self, policy_table):
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.draw_one_arrow(i, j, policy_table[i][j])

    def print_value_table(self, value_table):
        for i in range(WIDTH):
            for j in range(HEIGHT):
                self.text_value(i, j, value_table[i][j])

    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.rectangle)
        self.update()

    """
    버튼 evaluate
    """
    def evaluate_policy(self):
        self.evaluation_count += 1
        for i in self.texts:
            self.canvas.delete(i)
        self.agent.policy_evaluation()
        self.print_value_table(self.agent.value_table)

    """
    버튼 improve
    """
    def improve_policy(self):
        self.improvement_count += 1
        for i in self.arrows:
            self.canvas.delete(i)
        self.agent.policy_improvement()
        self.draw_from_policy(self.agent.policy_table)


class Env:
    """

    환경에 대한 정보

    """
    def __init__(self):
        self.transition_probability = TRANSITION_PROB   # 상태변환 확률
        self.possible_actions = POSSIBLE_ACTIONS        # 에이전트가 할 수 있는 모든 행동 인덱스 : 0=좌, 1=우, 2=하, 3=상
        self.width = WIDTH                              # 그리드월드 가로
        self.height = HEIGHT                            # 그리드월드 세로

        # reward - 2차원 리스트로 초기화
        # - 그리드월드 가로 x 그리드월드 세로 크기만큼
        # (1) [0] * 5 = [0, 0, 0, 0, 0]
        # (2) [0] * 5 for _ in range(5)] = [[0, 0, 0, 0, 0],
        #                                   [0, 0, 0, 0, 0],
        #                                   [0, 0, 0, 0, 0],
        #                                   [0, 0, 0, 0, 0],
        #                                   [0, 0, 0, 0, 0]]
        self.reward = [[0] * WIDTH for _ in range(HEIGHT)]
        self.reward[2][2] = 1   # (2,2) 좌표 동그라미 위치에 보상 1
        self.reward[1][2] = -1  # (1,2) 좌표 세모 위치에 보상 -1
        self.reward[2][1] = -1  # (2,1) 좌표 세모 위치에 보상 -1

        # state - 그리드월드 모든 좌표
        # # [[0, 0],
        #    [0, 1],
        #    [0, 2],
        #     ...  ,
        #    [4, 2],
        #    [4, 3],
        #    [4, 4]]
        self.all_state = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                state = [x, y]
                self.all_state.append(state)

    """
    (1) 환경이 주는 보상 
    ---------------
        :param 
            - state  : 현재 상태(좌표) 
            - action : 행동 index (0=좌,1=우,2=하,3=상)
        :return 현재 상태(state)에서 특정 행동(action) 취한 후에 얻게되는 보상(reward)
    """
    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    """
    (2) 환경이 주는 다음 상태
    ---------------
    :return 특정 상태(state)에서 특정 행동(action)을 취한 후에 에이전트가 가는 다음 상태(state)(좌표)
    """
    def state_after_action(self, state, action_index):
        # 좌표로 표현한 행동 : (좌,우,하,상) 중 index 에 해당하는 행동을 좌표로 나타낸다.
        action = ACTIONS[action_index]

        # 시작 state 에서 아래로 action 취한 후의 좌표
        # 예) 현재 state : [0, 0], action index 2 이므로 : [0, -1]
        #     -> state[0] + action[0] : 0 + 0 = 0
        #     -> state[1] + action[1] : 0 + (-1) = -1
        #       -> action을 취한 후의 state : [0, -1] 에 대해 boundary 확인하고 리턴한다.
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    """
    (3)
    (정적 메소드 선언)
    ---------------
        :return 그리드월드 벗어나지 않게 state 반환 
    """
    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1
                    if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT - 1
                    if state[1] > HEIGHT - 1 else state[1])
        return state

    def get_transition_prob(self, state, action):
        return self.transition_probability

    """
    (4) 환경이 주는 모든 상태 정보
    dynamic programming 에서는 에이전트가 '모든 상태'에 대해 벨만 방정식을 계산한다.
    ---------------
        :return 그리드월드에서의 모든 상태(좌표) 
    """
    def get_all_states(self):
        return self.all_state
