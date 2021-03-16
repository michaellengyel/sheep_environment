import environment
import cv2
import random


def main():

    print("Random Testing Suite Started...")

    # Create Environment
    env = environment.Environment("data/reduced_height_map.jpg", 15, 100)

    # Reset Environment
    env.reset()

    # Loop Environment
    while True:

        # Render map and sub-map
        env.render_map()
        env.render_sub_map()

        cv2.waitKey(1)

        # Generate a random number 0-7
        action = random.randrange(0, 8)

        next_state, reward, game_over = env.step(action)

        print("reward: {}".format(reward),
              "total reward: {}".format(env.agent_reward),
              "game_over: {}".format(game_over))


if __name__ == '__main__':
    main()
