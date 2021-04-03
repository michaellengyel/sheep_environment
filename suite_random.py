from environment import Environment
import cv2
import random


def main():

    print("Random Testing Suite Started...")

    # Create Environment
    env = Environment("data/map_small.jpg", 15, 100, 10, 100)

    # Reset Environment
    env.reset()

    # Loop Environment
    while not env.game_over:

        # Render map and sub-map
        env.render_map()
        env.render_sub_map()

        cv2.waitKey(1)

        # Generate a random number 0-7
        action = random.randrange(0, 8)

        next_state, reward, game_over = env.step(action)

        print("reward: {}".format(reward),
              "total reward: {}".format(env.agent_reward),
              "game_over: {}".format(game_over),
              "total_reward: {}".format(env.total_generated_rewards)
              )


if __name__ == '__main__':
    main()
