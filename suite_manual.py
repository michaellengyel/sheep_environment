from environment import Environment
import cv2


def main():

    print("Manual Testing Started...")
    print("Use Keys W, A, D, X(Y), Q, R, Z, C to navigate.")

    # Create Environment
    env = Environment("data/map_small_edge.jpg", 15, 100, 10, 1000)

    # Reset Environment
    env.reset()

    # Loop Environment
    while not env.game_over:

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
        elif key == ord('x') or key == ord('y'):
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
