import pygame
import yaml
import time
import random
import collections

class Snake:
    def __init__(self) -> None:
        self.direction: str = 'right'
        self.direction_map: dict = {'right': (1, 0), 'left': (-1, 0), 'up': (0, -1), 'down': (0, 1)}
        self.body: collections.deque = collections.deque([(0, 0)])

    def __len__(self) -> int:
        return len(self.body)

    def update_direction(self, direction: str):
        if direction not in self.direction_map.keys():
            raise ValueError(f'Invalid direction. Available directions: {self.direction_map.keys()}')
        self.direction = self.direction_map[direction]

    def update(self, eat: bool = False, direction: None):
        if direction and direction != self.direction:
            self.update_direction(direction)
        if eat:
            new_body = [self.body[-1][0] + self.direction[0], self.body[-1][1] + self.direction[1]]
            self.body.append(new_body)
        self.body.append(self.body.popleft())
        
class Food:
    def __init__(self) -> None:
        self.position = (0, 0)

    def generate(self, available_positions: list):
        self.position = random.choice(available_positions)
        return self.position

class GraphicsUserInterface:
    def __init__(self, config: dict) -> None:
        pygame.init()
        self.config = config
        self.color = self.config['color']
        self.initialize_screen()

    def initialize_screen(self):
        world = self.config['world']
        screen_height = int(world['rows'] * world['cell_size'])
        screen_width = int(world['columns'] * world['cell_size'])
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill(self.color['background'])
        pygame.display.set_caption(self.config['display_app_name'])

    def draw_world(self):
        world = self.config['world']
        for x in range(world['columns']):
            for y in range(world['rows']):
                self.draw_cell(x, y, self.color['grid'])

    def draw_snake(self, snake: list):
        x, y = snake[0]
        self.draw_cell(x, y, self.color['snake']['head'])
        for (_x, _y) in snake[1:]:
            if not abs(x - _x) <= 1 and not abs(y - _y) <= 1:
                raise ValueError('Snake body part is not adjacent to the previous part')
            x, y = _x, _y
            self.draw_cell(x, y, self.color['snake']['body'])

    def draw_cell(self, x: int, y: int, color: tuple, border=1) -> None:

        if x >= self.config['world']['columns'] or y >= self.config['world']['rows']:
            raise ValueError(f'Cell coordinates out of bounds. Requested: ({x}, {y}). Available: ({self.config["world"]["columns"]}, {self.config["world"]["rows"]})')

        cell_size = self.config['world']['cell_size']
        cell = (x*cell_size, y*cell_size, cell_size-border, cell_size-border)
        pygame.draw.rect(self.screen, color, cell)

class Game:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.snake = Snake()
        self.food = Food()
        self.gui = GraphicsUserInterface(config['gui'])

def main():
    settings = yaml.safe_load(open('settings.yaml'))
    
    game = Game(settings)
    game.gui.draw_world()
    while True:
        game.gui.draw_snake([(0, 0), (1, 0), (2, 0)])
        pygame.display.update()
        time.sleep(0.1)
    
    
    # # Main game loop
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    #     screen.fill((0, 0, 0))

    #     # Draw grid
    #     for x in range(0, screen.get_width(), grid_size):
    #         for y in range(0, screen.get_height(), grid_size):
    #             rect = pygame.Rect(x, y, grid_size, grid_size)
    #             pygame.draw.rect(screen, (40, 40, 40), rect, 1)

    #     pygame.display.flip()

    # pygame.quit()

if __name__ == '__main__':
    main()
