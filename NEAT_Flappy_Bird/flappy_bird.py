from dataclasses import dataclass
import pygame
import random
import os
from pathlib import Path
from typing import List, Tuple, Optional
import neat
from pygame.surface import Surface
from pygame.mask import Mask

# Initialize Pygame
pygame.init()
pygame.font.init()

# Game Constants
@dataclass(frozen=True)
class GameConfig:
    WIDTH: int = 600
    HEIGHT: int = 800
    FLOOR: int = 730
    FPS: int = 30
    PIPE_GAP: int = 200
    PIPE_VELOCITY: int = 5
    BIRD_MAX_ROTATION: int = 25
    BIRD_ROTATION_VELOCITY: int = 20
    BIRD_ANIMATION_TIME: int = 5
    BIRD_JUMP_VELOCITY: int = -10.5
    GRAVITY: float = 3.0

# Asset Management
class AssetLoader:
    def __init__(self) -> None:
        self.base_path = Path("imgs")
        self.pipe_img = self._load_and_scale("pipe.png", scale=2)
        self.bg_img = self._load_and_scale("bg.png", target_size=(600, 900))
        self.base_img = self._load_and_scale("base.png", scale=2)
        self.bird_imgs = [self._load_and_scale(f"bird{x}.png", scale=2) for x in range(1, 4)]
        
        # Fonts
        self.stat_font = pygame.font.SysFont("comicsans", 50)
        self.end_font = pygame.font.SysFont("comicsans", 70)

    def _load_and_scale(self, filename: str, scale: int = 1, target_size: Optional[Tuple[int, int]] = None) -> Surface:
        image = pygame.image.load(self.base_path / filename).convert_alpha()
        if target_size:
            return pygame.transform.scale(image, target_size)
        if scale != 1:
            return pygame.transform.scale2x(image)
        return image

class Bird:
    def __init__(self, x: int, y: int, assets: AssetLoader):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = y
        self.img_count = 0
        self.assets = assets
        self.current_img = assets.bird_imgs[0]

    def jump(self) -> None:
        self.velocity = GameConfig.BIRD_JUMP_VELOCITY
        self.tick_count = 0
        self.height = self.y

    def move(self) -> None:
        self.tick_count += 1
        displacement = (self.velocity * self.tick_count + 
                       0.5 * GameConfig.GRAVITY * self.tick_count ** 2)

        # Terminal velocity
        displacement = min(max(displacement, -16), 16)

        if displacement < 0:
            displacement -= 2

        self.y += displacement

        # Tilt handling
        if displacement < 0 or self.y < self.height + 50:
            self.tilt = min(GameConfig.BIRD_MAX_ROTATION, self.tilt + 2)
        else:
            if self.tilt > -90:
                self.tilt -= GameConfig.BIRD_ROTATION_VELOCITY

    def draw(self, window: Surface) -> None:
        self.img_count = (self.img_count + 1) % (GameConfig.BIRD_ANIMATION_TIME * 4)
        
        # Animation frames
        if self.tilt <= -80:
            self.current_img = self.assets.bird_imgs[1]
        else:
            frame_index = (self.img_count // GameConfig.BIRD_ANIMATION_TIME) % 3
            self.current_img = self.assets.bird_imgs[frame_index]

        self._blit_rotated(window)

    def _blit_rotated(self, surface: Surface) -> None:
        rotated_img = pygame.transform.rotate(self.current_img, self.tilt)
        new_rect = rotated_img.get_rect(center=self.current_img.get_rect(topleft=(self.x, self.y)).center)
        surface.blit(rotated_img, new_rect.topleft)

    def get_mask(self) -> Mask:
        return pygame.mask.from_surface(self.current_img)

class Pipe:
    def __init__(self, x: int, assets: AssetLoader):
        self.x = x
        self.assets = assets
        self.height = random.randrange(50, 450)
        self.top = self.height - assets.pipe_img.get_height()
        self.bottom = self.height + GameConfig.PIPE_GAP
        self.pipe_top = pygame.transform.flip(assets.pipe_img, False, True)
        self.pipe_bottom = assets.pipe_img
        self.passed = False

    def move(self) -> None:
        self.x -= GameConfig.PIPE_VELOCITY

    def draw(self, window: Surface) -> None:
        window.blit(self.pipe_top, (self.x, self.top))
        window.blit(self.pipe_bottom, (self.x, self.bottom))

    def collide(self, bird: Bird) -> bool:
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.pipe_top)
        bottom_mask = pygame.mask.from_surface(self.pipe_bottom)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        return bool(
            bird_mask.overlap(bottom_mask, bottom_offset) or 
            bird_mask.overlap(top_mask, top_offset)
        )

class Base:
    def __init__(self, y: int, assets: AssetLoader):
        self.y = y
        self.assets = assets
        self.width = assets.base_img.get_width()
        self.x1 = 0
        self.x2 = self.width

    def move(self) -> None:
        self.x1 -= GameConfig.PIPE_VELOCITY
        self.x2 -= GameConfig.PIPE_VELOCITY

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, window: Surface) -> None:
        window.blit(self.assets.base_img, (self.x1, self.y))
        window.blit(self.assets.base_img, (self.x2, self.y))

class FlappyBirdGame:
    def __init__(self):
        self.window = pygame.display.set_mode((GameConfig.WIDTH, GameConfig.HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.assets = AssetLoader()
        self.clock = pygame.time.Clock()
        self.generation = 0
        self.draw_lines = False

    def eval_genomes(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
        self.generation += 1

        networks = []
        birds = []
        ge = []
        MAX_SCORE=200

        for _, genome in genomes:
            genome.fitness = 0
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            networks.append(network)
            birds.append(Bird(230, 350, self.assets))
            ge.append(genome)

        base = Base(GameConfig.FLOOR, self.assets)
        pipes = [Pipe(700, self.assets)]
        score = 0

        running = True
        while running and birds:
            self.clock.tick(GameConfig.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            pipe_ind = 0
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipe_top.get_width():
                    pipe_ind = 1

            # Update bird positions and neural networks
            for x, bird in enumerate(birds):
                ge[x].fitness += 0.1
                bird.move()

                output = networks[x].activate((
                    bird.y,
                    abs(bird.y - pipes[pipe_ind].height),
                    abs(bird.y - pipes[pipe_ind].bottom)
                ))

                if output[0] > 0.5:
                    bird.jump()

            # Update game objects
            base.move()
            rem = []
            add_pipe = False

            for pipe in pipes:
                pipe.move()

                # Collision detection
                for bird in birds[:]:  # Create a copy of the list for safe removal
                    if pipe.collide(bird):
                        idx = birds.index(bird)
                        ge[idx].fitness -= 1
                        networks.pop(idx)
                        ge.pop(idx)
                        birds.remove(bird)

                if pipe.x + pipe.pipe_top.get_width() < 0:
                    rem.append(pipe)

                if not pipe.passed and pipe.x < birds[0].x if birds else 0:
                    pipe.passed = True
                    add_pipe = True

            if add_pipe:
                score += 1
                for genome in ge:
                    genome.fitness += 5
                pipes.append(Pipe(GameConfig.WIDTH, self.assets))

            for r in rem:
                pipes.remove(r)

            # Check if birds hit the ground or go too high
            for bird in birds[:]:
                if bird.y + bird.current_img.get_height() >= GameConfig.FLOOR or bird.y < -50:
                    idx = birds.index(bird)
                    birds.pop(idx)
                    networks.pop(idx)
                    ge.pop(idx)

            self.draw_game(birds, pipes, base, score, pipe_ind)

            if score >= MAX_SCORE:
                print(f"Reached maximum score of {MAX_SCORE}")
                running = False
                break

    def draw_game(self, birds: List[Bird], pipes: List[Pipe], base: Base, 
                  score: int, pipe_ind: int) -> None:
        self.window.blit(self.assets.bg_img, (0, 0))

        for pipe in pipes:
            pipe.draw(self.window)

        base.draw(self.window)
        
        for bird in birds:
            if self.draw_lines:
                try:
                    pygame.draw.line(
                        self.window,
                        (255, 0, 0),
                        (bird.x + bird.current_img.get_width()/2, bird.y + bird.current_img.get_height()/2),
                        (pipes[pipe_ind].x + pipes[pipe_ind].pipe_top.get_width()/2, pipes[pipe_ind].height),
                        5
                    )
                    pygame.draw.line(
                        self.window,
                        (255, 0, 0),
                        (bird.x + bird.current_img.get_width()/2, bird.y + bird.current_img.get_height()/2),
                        (pipes[pipe_ind].x + pipes[pipe_ind].pipe_bottom.get_width()/2, pipes[pipe_ind].bottom),
                        5
                    )
                except IndexError:
                    pass
            bird.draw(self.window)

        # Draw stats
        stats = [
            f"Score: {score}",
            f"Gen: {self.generation}",
            f"Alive: {len(birds)}"
        ]

        
    
        for i, stat in enumerate(stats):
            text = self.assets.stat_font.render(stat, True, (255, 255, 255))
            x = 10 if i > 0 else GameConfig.WIDTH - text.get_width() - 15
            y = 10 + (i * 40)
            self.window.blit(text, (x, y))

        pygame.display.update()
        
import joblib 
def save_model(winner, filename="winner_genome.joblib"):
    with open(filename, "wb") as f:
        joblib.dump(winner, f)
    print(f"Model saved to {filename}")

def run_neat(config_path: str) -> None:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    game = FlappyBirdGame()
    winner = population.run(game.eval_genomes, 20)
    print(f'\nBest genome:\n{winner}')
    
    save_model(winner)


if __name__ == '__main__':
    local_dir = Path(__file__).parent
    config_path = local_dir / 'config-feedforward.txt'
    run_neat(str(config_path))