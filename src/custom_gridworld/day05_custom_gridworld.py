class CustomGridWorld:
    """
    A simple custom Grid World environment.

    Grid layout:

    S . . .
    . W . .
    . . W .
    . . . G

    S = Start
    W = Wall
    G = Goal
    A = Agent
    """

    def __init__(self):
        self.grid_size = 4

        self.start_position = (0, 0)
        self.goal_position = (3, 3)
        self.wall_positions = {(1, 1), (2, 2)}

        self.agent_position = self.start_position
        self.step_count = 0
        self.max_steps = 50

        self.actions = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
        }

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.agent_position = self.start_position
        self.step_count = 0
        return self.agent_position

    def step(self, action):
        """
        Apply an action and return next_state, reward, and done.
        """
        if action not in self.actions:
            raise ValueError("Invalid action. Action must be 0, 1, 2, or 3.")

        self.step_count += 1

        current_row, current_col = self.agent_position
        next_row, next_col = current_row, current_col

        if action == 0:      # Up
            next_row -= 1
        elif action == 1:    # Down
            next_row += 1
        elif action == 2:    # Left
            next_col -= 1
        elif action == 3:    # Right
            next_col += 1

        next_position = (next_row, next_col)

        if not self._is_inside_grid(next_position):
            reward = -1
            done = False
            next_position = self.agent_position

        elif next_position in self.wall_positions:
            reward = -1
            done = False
            next_position = self.agent_position

        elif next_position == self.goal_position:
            reward = 10
            done = True
            self.agent_position = next_position

        else:
            reward = -0.1
            done = False
            self.agent_position = next_position

        if self.step_count >= self.max_steps:
            done = True

        return self.agent_position, reward, done

    def render(self):
        """
        Print the current Grid World state.
        """
        for row in range(self.grid_size):
            row_display = []

            for col in range(self.grid_size):
                position = (row, col)

                if position == self.agent_position:
                    row_display.append("A")
                elif position == self.start_position:
                    row_display.append("S")
                elif position == self.goal_position:
                    row_display.append("G")
                elif position in self.wall_positions:
                    row_display.append("W")
                else:
                    row_display.append(".")

            print(" ".join(row_display))

        print()

    def _is_inside_grid(self, position):
        """
        Check whether the position is inside the grid.
        """
        row, col = position

        return (
            0 <= row < self.grid_size
            and 0 <= col < self.grid_size
        )


def run_manual_demo():
    env = CustomGridWorld()

    state = env.reset()

    print("=== Custom Grid World Demo ===")
    print(f"Initial State: {state}")
    env.render()

    actions = [
        3,  # Right
        3,  # Right
        1,  # Down
        1,  # Down
        3,  # Right
        1,  # Down
    ]

    for step_number, action in enumerate(actions, start=1):
        next_state, reward, done = env.step(action)

        print(f"Step {step_number}")
        print(f"Action: {action} ({env.actions[action]})")
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        env.render()

        if done:
            print("Episode finished.")
            break


if __name__ == "__main__":
    run_manual_demo()