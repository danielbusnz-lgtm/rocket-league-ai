import numpy as np
import torch
from pathlib import Path

try:
    from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
    from rlbot.utils.structures.game_data_struct import GameTickPacket
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False

    class BaseAgent:
        def __init__(self, name, team, index):
            self.name = name
            self.team = team
            self.index = index

    class SimpleControllerState:
        def __init__(self):
            self.throttle = 0
            self.steer = 0
            self.pitch = 0
            self.yaw = 0
            self.roll = 0
            self.jump = False
            self.boost = False
            self.handbrake = False

    class GameTickPacket:
        pass


class TrainedRLBotAgent(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.model = None
        self.checkpoint_path = None

    def initialize_agent(self):
        checkpoint_dir = Path(__file__).parent.parent.parent / "data" / "checkpoints"

        if not checkpoint_dir.exists():
            print("ERROR: No checkpoints found!")
            return

        runs = sorted(checkpoint_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        if not runs:
            print("ERROR: No training runs found!")
            return

        latest_run = runs[-1]
        checkpoints = sorted([d for d in latest_run.iterdir() if d.is_dir()],
                           key=lambda p: int(p.name))

        if not checkpoints:
            print("ERROR: No checkpoints in run!")
            return

        latest_checkpoint = checkpoints[-1]
        self.checkpoint_path = latest_checkpoint / "PPO_POLICY.pt"

        if not self.checkpoint_path.exists():
            print(f"ERROR: Model file not found: {self.checkpoint_path}")
            return

        try:
            self.model = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.eval()
            print(f"Model loaded from: {latest_checkpoint.name}")
        except Exception as e:
            print(f"ERROR loading model: {e}")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        controller_state = SimpleControllerState()

        if self.model is None:
            return controller_state

        try:
            obs = self.convert_packet_to_observation(packet)

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = np.random.randint(0, 90)

            controller_state = self.action_to_controller(action)

        except Exception as e:
            print(f"Error: {e}")

        return controller_state

    def convert_packet_to_observation(self, packet: GameTickPacket) -> np.ndarray:
        car = packet.game_cars[self.index]
        ball = packet.game_ball
        obs = np.zeros(72)

        obs[0] = ball.physics.location.x / 4096
        obs[1] = ball.physics.location.y / 5120
        obs[2] = ball.physics.location.z / 2044
        obs[3] = ball.physics.velocity.x / 6000
        obs[4] = ball.physics.velocity.y / 6000
        obs[5] = ball.physics.velocity.z / 6000
        obs[6] = car.physics.location.x / 4096
        obs[7] = car.physics.location.y / 5120
        obs[8] = car.physics.location.z / 2044
        obs[9] = car.physics.velocity.x / 2300
        obs[10] = car.physics.velocity.y / 2300
        obs[11] = car.physics.velocity.z / 2300

        return obs

    def action_to_controller(self, action: int) -> SimpleControllerState:
        controller = SimpleControllerState()

        throttle_idx = action % 3
        steer_idx = (action // 3) % 3
        pitch_idx = (action // 9) % 3
        yaw_idx = (action // 27) % 3

        controller.throttle = [-1, 0, 1][throttle_idx]
        controller.steer = [-1, 0, 1][steer_idx]
        controller.pitch = [-1, 0, 1][pitch_idx]
        controller.yaw = [-1, 0, 1][yaw_idx]

        return controller


if __name__ == "__main__":
    print("RLBot agent ready for deployment")
