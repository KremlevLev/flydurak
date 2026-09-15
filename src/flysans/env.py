from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnvConfig:
    batch_size: int = 64
    horizon: int = 180
    speed: float = 0.065
    heart_radius: float = 0.045
    fps: int = 30
    curriculum_level: int = 0
    forced_attack: int | None = None


class SansFightEnv:
    """Clean-room, vectorized Sans-style survival curriculum with no renderer."""
    observation_size = 16
    action_size = 5
    attack_names = ("vertical_gap", "horizontal_gap", "bone_rain", "blaster", "blue_slam")

    def __init__(self, config: EnvConfig, device: torch.device | str = "cpu"):
        self.config, self.device = config, torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.reset(0)

    def reset(self, seed: int = 0):
        self.generator.manual_seed(seed)
        b = self.config.batch_size
        level = min(self.config.curriculum_level, len(self.attack_names) - 1)
        if self.config.forced_attack is None:
            self.attack = torch.randint(level + 1, (b,), generator=self.generator, device=self.device)
        else:
            self.attack = torch.full((b,), self.config.forced_attack, dtype=torch.long, device=self.device)
        self.heart = torch.empty(b, 2, device=self.device).uniform_(-0.15, 0.15, generator=self.generator)
        self.velocity = torch.zeros(b, 2, device=self.device)
        self.gap = torch.empty(b, device=self.device).uniform_(-0.55, 0.55, generator=self.generator)
        self.variant = torch.empty(b, device=self.device).uniform_(0.0, 1.0, generator=self.generator)
        self.t = torch.zeros(b, dtype=torch.long, device=self.device)
        self.alive = torch.ones(b, dtype=torch.bool, device=self.device)
        return self.observe()

    def _phase(self): return self.t.float() / self.config.horizon

    def _features(self):
        sweep = 1.2 - 2.4 * self._phase()
        rain_x = 0.70 * torch.sin(self.t.float() * 0.07 + self.variant * 6.283185)
        telegraph = ((self.t % 45) >= 32).float()
        axis = torch.where(self.variant > 0.5, self.heart[:, 0], self.heart[:, 1])
        return sweep, rain_x, telegraph, axis

    def observe(self):
        sweep, rain_x, telegraph, axis = self._features()
        one_hot = torch.nn.functional.one_hot(self.attack, len(self.attack_names)).float()
        return torch.cat((self.heart, self.velocity, sweep[:, None], self.gap[:, None], self._phase()[:, None], rain_x[:, None], telegraph[:, None], axis[:, None], one_hot, self.variant[:, None]), 1)

    def _collision(self):
        x, y = self.heart[:, 0], self.heart[:, 1]
        sweep, rain_x, _, _ = self._features()
        r = self.config.heart_radius
        vertical = ((x - sweep).abs() < 0.075 + r) & ((y - self.gap).abs() > 0.19 - r)
        horizontal = ((y - sweep).abs() < 0.075 + r) & ((x - self.gap).abs() > 0.19 - r)
        rain = (self.t > 12) & ((x - rain_x).abs() < 0.055 + r)
        beam_live = ((self.t % 45) >= 38) & ((self.t % 45) < 44)
        beam_distance = torch.where(self.variant > 0.5, (y - self.gap).abs(), (x - self.gap).abs())
        blaster = beam_live & (beam_distance < 0.11 + r)
        floor_bones = (y < -0.72) & (torch.sin(x * 15.0 + self.t.float() * 0.18) > 0.25)
        return torch.stack((vertical, horizontal, rain, blaster, floor_bones)).gather(0, self.attack[None]).squeeze(0)

    def step(self, action: torch.Tensor):
        if action.shape != (self.config.batch_size,): raise ValueError("action must have shape [batch_size]")
        if bool(((action < 0) | (action >= 5)).any()): raise ValueError("action values must be in [0, 4]")
        vertical_distance = (self.heart[:, 1] - self.gap).abs()
        horizontal_distance = (self.heart[:, 0] - self.gap).abs()
        old_gap_distance = torch.where(self.attack == 1, horizontal_distance, vertical_distance)
        _, rain_x, telegraph, _ = self._features()
        old_rain_distance = (self.heart[:, 0] - rain_x).abs()
        old_beam_distance = torch.where(self.variant > 0.5, vertical_distance, horizontal_distance)
        delta = torch.tensor([[0., 1.], [0., -1.], [-1., 0.], [1., 0.], [0., 0.]], device=self.device)[action]
        blue = self.attack == 4
        self.velocity = torch.where(blue[:, None], self.velocity + torch.tensor([0., -0.018], device=self.device), delta * self.config.speed)
        jump = blue & (action == 0) & (self.heart[:, 1] <= -0.50)
        self.velocity[jump, 1] = 0.24
        self.velocity.clamp_(-0.25, 0.25)
        self.heart = torch.clamp(self.heart + self.velocity, -1.0, 1.0)
        hit = self.alive & self._collision()
        vertical_distance = (self.heart[:, 1] - self.gap).abs()
        horizontal_distance = (self.heart[:, 0] - self.gap).abs()
        new_gap_distance = torch.where(self.attack == 1, horizontal_distance, vertical_distance)
        gap_progress = old_gap_distance - new_gap_distance
        gap_attack = (self.attack == 0) | (self.attack == 1)
        shaping = torch.where(gap_attack, 0.5 * gap_progress, torch.zeros_like(gap_progress))
        aligned = gap_attack.float() * (new_gap_distance < 0.12).float() * 0.002
        new_rain_distance = (self.heart[:, 0] - rain_x).abs()
        shaping += (self.attack == 2).float() * 0.25 * torch.clamp(new_rain_distance - old_rain_distance, -0.08, 0.08)
        new_beam_distance = torch.where(self.variant > 0.5, vertical_distance, horizontal_distance)
        shaping += (self.attack == 3).float() * telegraph * 0.25 * torch.clamp(new_beam_distance - old_beam_distance, -0.08, 0.08)
        shaping += (self.attack == 4).float() * (self.heart[:, 1] < -0.45).float() * torch.clamp(self.velocity[:, 1], min=0.0) * 0.05
        reward = self.alive.float() / self.config.horizon + shaping + aligned - hit.float()
        self.alive &= ~hit
        self.t += 1
        timeout = self.t >= self.config.horizon
        done = ~self.alive | timeout
        return self.observe(), reward, done, {"hit": hit, "survived": timeout & self.alive, "attack": self.attack}

    def oracle_action(self):
        action = torch.full((self.config.batch_size,), 4, dtype=torch.long, device=self.device)
        vertical = self.attack == 0
        action[vertical & (self.heart[:, 1] < self.gap - 0.04)] = 0
        action[vertical & (self.heart[:, 1] > self.gap + 0.04)] = 1
        horizontal = self.attack == 1
        action[horizontal & (self.heart[:, 0] < self.gap - 0.04)] = 3
        action[horizontal & (self.heart[:, 0] > self.gap + 0.04)] = 2
        _, rain_x, telegraph, _ = self._features()
        rain = self.attack == 2
        rain_target = torch.clamp(rain_x + 0.35, -0.90, 0.90)
        action[rain & (self.heart[:, 0] < rain_target - 0.04)] = 3
        action[rain & (self.heart[:, 0] > rain_target + 0.04)] = 2
        blaster = (self.attack == 3) & (telegraph > 0)
        beam_target = torch.where(self.gap > 0, torch.full_like(self.gap, -0.85), torch.full_like(self.gap, 0.85))
        vertical_beam = blaster & (self.variant > 0.5)
        action[vertical_beam & (self.heart[:, 1] < beam_target - 0.04)] = 0
        action[vertical_beam & (self.heart[:, 1] > beam_target + 0.04)] = 1
        horizontal_beam = blaster & (self.variant <= 0.5)
        action[horizontal_beam & (self.heart[:, 0] < beam_target - 0.04)] = 3
        action[horizontal_beam & (self.heart[:, 0] > beam_target + 0.04)] = 2
        blue = self.attack == 4
        action[blue & (self.heart[:, 1] < -0.35)] = 0
        return action


BoneSweepEnv = SansFightEnv
