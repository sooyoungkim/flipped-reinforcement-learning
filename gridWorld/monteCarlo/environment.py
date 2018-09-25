import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100      # 픽셀 수
HEIGHT = 5      # 그리드월드 세로
WIDTH = 5       # 그리드월드 가로


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']    # 상(up), 하(down), 좌(left), 우(right)
        self.n_actions = len(self.action_space)
        self.title('monte carlo')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # 그리드 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1])
        self.triangle2 = canvas.create_image(150, 250, image=self.shapes[1])
        self.circle = canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()

        return canvas

    @staticmethod
    def load_images():
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    """
    (정적 메소드 선언)
    ---------------
        :return 좌표 정보를 state 로 반환
    """
    @staticmethod
    def coords_to_state(coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()

        # action 에 따라 current state 에서 next state 까지 이동해야 할 크기 계산
        if action == 0:    # 상
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 하
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 좌
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # 우
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        # 에이전트(빨간 네모) 이동 : 현재 state 에서 base_action 만큼 이동
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # 에이전트(빨간 네모)를 맨 앞으로 배치 (다른 것에의해 가리지 않도록)
        self.canvas.tag_raise(self.rectangle)
        # 이동한 좌표 정보 얻기
        next_state = self.canvas.coords(self.rectangle)

        # 보상(reward) 계산
        if next_state == self.canvas.coords(self.circle):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.triangle1), self.canvas.coords(self.triangle2)]:
            reward = -100
            done = True
        else:
            reward = 0
            done = False

        # 이동한 좌표 정보를 state 로 얻기
        next_state = self.coords_to_state(next_state)

        # 상태(state)는 리스트, 보상(reward)은 숫자, 완료 여부(done)는 boolean
        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()

