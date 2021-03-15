import random
import cv2


class Environment:

    # Initialize environment and sets agent to random location (s0, a0)
    def __init__(self, map_img_path, fov, food_spawn_threshold):
        # Private input variables
        self.map_img_path = map_img_path
        self.fov = fov
        self.food_spawn_threshold = food_spawn_threshold

        # Private variables
        self.offset = int((self.fov - 1) / 2)
        self.map_img_agents = self.load_map()
        self.map_img_calculations = self.load_map()
        self.env_width = self.map_img_agents.shape[0]
        self.env_height = self.map_img_agents.shape[1]
        self.agent_x = 0
        self.agent_y = 0
        self.agent_reward = 0
        self.agent_current_reward = 0
        self.total_generated_rewards = 0
        self.game_over = False
        self.reward_value = 1
        self.step_cost = 0

        self.init_map()  # s0
        self.init_agent_pos()  # a0

        cv2.namedWindow('MAP', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MAP', 900, 900)
        cv2.namedWindow('SUBMAP', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SUBMAP', 400, 400)

    '''Environment Initialization Functions'''

    # Initialize position of agent randomly (a0)
    def init_agent_pos(self):
        rand_x = random.randrange(self.offset, self.env_width - self.offset)
        rand_y = random.randrange(self.offset, self.env_height - self.offset)
        self.agent_x = rand_x
        self.agent_y = rand_y
        self.map_img_agents[self.agent_x, self.agent_y, 0] = 255
        self.map_img_agents[self.agent_x, self.agent_y, 1] = 0
        self.map_img_agents[self.agent_x, self.agent_y, 2] = 0

    # Load the map from a .png or .jpg
    def load_map(self):
        img = cv2.imread(self.map_img_path, cv2.IMREAD_COLOR)
        map_img = img.copy()
        return map_img

    # Spawn grass
    def init_map(self):

        # Iterate through x dimension
        for i in range(self.offset, self.env_width - self.offset):
            # Iterate through y dimension
            for j in range(self.offset, self.env_height - self.offset):
                # Generate a number in the range of the pixel at i, j
                rand = random.randrange(0, self.map_img_agents[i, j, 0])
                # If the generated number is 0 and pixel i, j is sub 30
                if (self.map_img_agents[i, j, 0] < self.food_spawn_threshold) & (rand <= 3):  # TODO (rand == 0)
                    # Set pixel red
                    self.map_img_agents[i, j, 0] = 0
                    self.map_img_agents[i, j, 1] = 255
                    self.map_img_agents[i, j, 2] = 0
                    # Increment the number of rewards counter
                    self.total_generated_rewards += 1

    # Reset Function
    def reset(self):
        self.map_img_agents = self.load_map()
        self.map_img_calculations = self.load_map()
        self.init_agent_pos()
        self.init_map()

        self.sub_map_img = self.map_img_agents[self.agent_x - self.offset:self.agent_x + self.offset + 1,
                           self.agent_y - self.offset:self.agent_y + self.offset + 1, :]

        return self.sub_map_img

    '''Getter Function'''

    '''Debug Helper Functions'''

    # Might need to make a torch.tensor from the numpy array
    def make_tensor_from_image(self):
        print("Making torch.tensor from image")

    def print_map_info(self):
        print(type(self.map_img_agents))
        print(self.map_img_agents.shape)

    def print_agent_info(self):
        print(self.agent_y, self.agent_x)

    '''GUI Render Functions'''

    def render_map(self):
        cv2.imshow('MAP', self.map_img_agents)

    # Render the map of the environment each tick
    def render_sub_map(self):
        cv2.imshow('SUBMAP', self.sub_map_img)

    '''Environment Logic Functions'''

    def movement_decision(self, x, y):

        lower_inside_bounds_x = ((self.agent_x + x) > self.offset)
        lower_inside_bounds_y = ((self.agent_y + y) > self.offset)
        upper_inside_bounds_x = ((self.agent_x + x) < self.env_width - self.offset)
        upper_inside_bounds_y = ((self.agent_y + y) < self.env_height - self.offset)

        is_within_bounds = lower_inside_bounds_x & lower_inside_bounds_y & upper_inside_bounds_x & upper_inside_bounds_y

        if is_within_bounds:
            # Un-paint the agent
            self.map_img_agents[self.agent_x, self.agent_y, 0] = self.map_img_calculations[
                self.agent_x, self.agent_y, 0]
            self.map_img_agents[self.agent_x, self.agent_y, 1] = self.map_img_calculations[
                self.agent_x, self.agent_y, 1]
            self.map_img_agents[self.agent_x, self.agent_y, 2] = self.map_img_calculations[
                self.agent_x, self.agent_y, 2]
            # Update the agent's position
            self.agent_x += x
            self.agent_y += y
            # Paint the agent to it's new position
            self.map_img_agents[self.agent_x, self.agent_y, 0] = 255
            self.map_img_agents[self.agent_x, self.agent_y, 1] = 0
            self.map_img_agents[self.agent_x, self.agent_y, 2] = 0
        # else:
        # print("Out of bounds")

    # Calculating net reward
    def calculate_reward(self, x, y):
        # Logic for gaining rewards
        if (self.map_img_agents[self.agent_x + x, self.agent_y + y, 1]) == 255:
            self.agent_reward += self.reward_value
        else:
            self.agent_reward += self.step_cost

    # Calculating step's reward
    def calculate_current_reward(self, x, y):
        # Logic for gaining rewards
        # Reset current reward to 0
        self.agent_current_reward = 0
        # If stepped on reward set current reward to reward_value
        if (self.map_img_agents[self.agent_x + x, self.agent_y + y, 1]) == 255:
            self.agent_current_reward = self.reward_value
        # If not stepped on reward set current reward to step_cost
        else:
            self.agent_current_reward = self.step_cost

    # Update the environment based on the action
    def step(self, action):

        if action == 0:  # UP
            self.calculate_reward(-1, 0)
            self.calculate_current_reward(-1, 0)
            self.movement_decision(-1, 0)
        elif action == 1:  # DOWN
            self.calculate_reward(1, 0)
            self.calculate_current_reward(1, 0)
            self.movement_decision(1, 0)
        elif action == 2:  # LEFT
            self.calculate_reward(0, -1)
            self.calculate_current_reward(0, -1)
            self.movement_decision(0, -1)
        elif action == 3:  # RIGHT
            self.calculate_reward(0, 1)
            self.calculate_current_reward(0, 1)
            self.movement_decision(0, 1)
        elif action == 4:  # UP/LEFT
            self.calculate_reward(-1, -1)
            self.calculate_current_reward(-1, -1)
            self.movement_decision(-1, -1)
        elif action == 5:  # UP/RIGHT
            self.calculate_reward(-1, 1)
            self.calculate_current_reward(-1, 1)
            self.movement_decision(-1, 1)
        elif action == 6:  # DOWN/LEFT
            self.calculate_reward(1, -1)
            self.calculate_current_reward(1, -1)
            self.movement_decision(1, -1)
        elif action == 7:  # DOWN/RIGHT
            self.calculate_reward(1, 1)
            self.calculate_current_reward(1, 1)
            self.movement_decision(1, 1)

        # Check if all rewards have been found, if yes set done to true
        if self.total_generated_rewards <= self.agent_reward:
            self.game_over = True

        # Crop the sub-map from the map
        self.sub_map_img = self.map_img_agents[self.agent_x - self.offset:self.agent_x + self.offset + 1,
                           self.agent_y - self.offset:self.agent_y + self.offset + 1, :]

        return self.sub_map_img, self.agent_current_reward, self.game_over


def main():
    print("Manual Testing Started...")
    print("Use Keys W, A, D, X, Q, R, Z, C to navigate.")

    # Create Environment
    env = Environment("data/reduced_height_map.jpg", 15, 100)

    # Reset Environment
    env.reset()

    # Loop Environment
    while True:

        # Render map and sub-map
        env.render_map()
        env.render_sub_map()

        key = cv2.waitKey(0)
        movement = ""
        action = 99

        if key == ord('p'):
            break
        elif key == ord('w'):
            action = 0
            movement += "UP"
        elif key == ord('x'):
            action = 1
            movement += "DOWN"
        elif key == ord('a'):
            action = 2
            movement += "LEFT"
        elif key == ord('d'):
            action = 3
            movement += "RIGHT"
        elif key == ord('q'):
            action = 4
            movement += "UP/LEFT"
        elif key == ord('e'):
            action = 5
            movement += "UP/RIGHT"
        elif key == ord('z'):
            action = 6
            movement += "DOWN/LEFT"
        elif key == ord('c'):
            action = 7
            movement += "DOWN/RIGHT"
        elif key:
            print("Invalid Input!")
            continue

        next_state, reward, game_over = env.step(action)

        print("reward: {}".format(reward),
              "total reward: {}".format(env.agent_reward),
              "game_over: {}".format(game_over),
              "Movement: {}".format(movement))


if __name__ == '__main__':
    main()
