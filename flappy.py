import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from os.path import join

class FlappyBirdEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super().__init__()
        pygame.init()

        # Kích thước màn hình
        self.screen_width = 400
        self.screen_height = 600

        # Hành động: 0 = không làm gì, 1 = bấm phím
        self.action_space = spaces.Discrete(2)

        # Trạng thái: bird_y, bird_speed, pipe_x, pipe_gap_y
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 0, 0]),
            high=np.array([self.screen_height, 20, self.screen_width, self.screen_height]),
            dtype=np.float32
        )

        # Pygame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Các biến tài nguyên
        self.background = None
        self.bird_images = []
        self.pipe_image = None

        # Cài đặt thông số trò chơi
        self.gravity = 2.5
        self.bird_speed = 0
        self.pipe_width = 80
        self.pipe_gap = 150
        self.speed = 15

        # Reset các biến của trò chơi
        self.bird_y = None
        self.pipe_x = None
        self.pipe_gap_y = None
        self.frame = 0

        #Điểm
        self.score = 0
        self.passed_pipe = False

        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self._load_assets()

    def _load_assets(self):
        """Tải tài nguyên."""
        self.background = pygame.image.load(join("assets", "sprites", "background-day.png")).convert_alpha()
        self.background = pygame.transform.scale(self.background, (self.screen_width, self.screen_height))
        self.bird_images = [
            pygame.image.load(join("assets", "sprites", "bluebird-upflap.png")).convert_alpha(),
            pygame.image.load(join("assets", "sprites", "bluebird-midflap.png")).convert_alpha(),
            pygame.image.load(join("assets", "sprites", "bluebird-downflap.png")).convert_alpha()
        ]
        self.pipe_image = pygame.image.load(join("assets", "sprites", "pipe-green.png")).convert_alpha()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.render_mode == "human" and not self.background:
            self._load_assets()

        # Khởi tạo vị trí chim và ống
        self.bird_y = self.screen_height // 2
        self.bird_speed = 0
        self.pipe_x = self.screen_width
        self.pipe_gap_y = random.randint(100, self.screen_height - self.pipe_gap - 100)

        # Reset điểm khi bắt đầu lại
        self.score = 0

        # Trả về trạng thái ban đầu
        state = self._get_state()
        if self.render_mode == "human":
            self._render_frame()
        return state, {}

    def step(self, action):
        # Thực hiện hành động
        if action == 1:  # Chim bay lên
            self.bird_speed = -10

        # Cập nhật trạng thái chim
        self.bird_speed += self.gravity
        self.bird_y += self.bird_speed

        # Di chuyển ống
        self.pipe_x -= self.speed

        # Nếu ống đi ra khỏi màn hình, tạo ống mới
        if self.pipe_x + self.pipe_width < 0:
            self.pipe_x = self.screen_width
            self.pipe_gap_y = random.randint(100, self.screen_height - self.pipe_gap - 100)
            self.passed_pipe = False  # Đặt lại khi tạo ống mới

        # Kiểm tra va chạm
        done = self._check_collision()
        reward = 0

        if not done:
            if not self.passed_pipe and self.pipe_x + self.pipe_width < self.screen_width // 6:
                self.score += 1
                self.passed_pipe = True  # Đánh dấu là đã qua ống
                reward = 1  # Thưởng nếu vượt qua ống

        # Trả về trạng thái tiếp theo, phần thưởng, và trạng thái kết thúc
        state = self._get_state()
        if self.render_mode == "human":
            self._render_frame()
        return state, reward, done, False, {}

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.screen is not None:
            pygame.quit()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self._load_assets()

        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)

        # Vẽ nền
        self.screen.blit(self.background, (0, 0))

        # Vẽ chim
        bird_image = self.bird_images[self.frame // 5]
        self.frame = (self.frame + 1) % 15
        self.screen.blit(bird_image, (self.screen_width // 6, self.bird_y))

        # Vẽ ống
        pipe_top = pygame.transform.flip(self.pipe_image, False, True)
        self.screen.blit(pipe_top, (self.pipe_x, self.pipe_gap_y - self.pipe_image.get_height()))
        self.screen.blit(self.pipe_image, (self.pipe_x, self.pipe_gap_y + self.pipe_gap))

        score_surface = self.font.render(f"{self.score}", True, (255, 255, 255))
        self.screen.blit(score_surface, ((self.screen_width - score_surface.get_width()) //2, 10))  # Vẽ điểm ở góc trên bên trái

        # Cập nhật màn hình
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _get_state(self):
        return np.array([
            self.bird_y,
            self.bird_speed,
            self.pipe_x,
            self.pipe_gap_y
        ], dtype=np.float32)

    def _check_collision(self):
        # Chim chạm đất hoặc vượt giới hạn
        if self.bird_y < 0 or self.bird_y > self.screen_height:
            return True

        # Chim chạm ống
        bird_x = self.screen_width // 6
        if self.pipe_x < bird_x < self.pipe_x + self.pipe_width:
            if not (self.pipe_gap_y < self.bird_y < self.pipe_gap_y + self.pipe_gap):
                return True

        return False
