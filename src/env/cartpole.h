#pragma once

#include <af/array.h>

/*
Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity     -Inf                    Inf

Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right

Reward:
    Reward is 1 for every step taken, including the termination step

Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]

Episode Termination
    Pole Angle is more than ±12°
    Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
    Episode length is greater than 200 (500 for v1).

Solved Requirements
    Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
*/

class CartPoleEnv
{
public:
    CartPoleEnv()
    {
        reset();
    }

    int data_space()
    {
        return 4;
    }

    int action_space()
    {
        return 2;
    }

    af::array reset()
    {
        std::default_random_engine rnd;
        state[0] = std::uniform_real_distribution<float>(-0.05, 0.05)(rnd);
        state[1] = std::uniform_real_distribution<float>(-0.05, 0.05)(rnd);
        state[2] = std::uniform_real_distribution<float>(-0.05, 0.05)(rnd);
        state[3] = std::uniform_real_distribution<float>(-0.05, 0.05)(rnd);
        steps_beyond_done = -1;
        step_ = 0;
        return af::array({4,1}, state);
    }

    std::tuple<af::array, float, bool> step(int action)
    {
        auto x = state[0];
        auto x_dot = state[1];
        auto theta = state[2];
        auto theta_dot = state[3];

        auto force = (action == 1) ? force_mag : -force_mag;
        auto costheta = std::cos(theta);
        auto sintheta = std::sin(theta);
        auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        auto thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        auto xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        x = x + tau * x_dot;
        x_dot = x_dot + tau * xacc;
        theta = theta + tau * theta_dot;
        theta_dot = theta_dot + tau * thetaacc;

        state[0] = x;
        state[1] = x_dot;
        state[2] = theta;
        state[3] = theta_dot;

        done = x < -x_threshold || x > x_threshold ||
               theta < -theta_threshold_radians || theta > theta_threshold_radians ||
               step_ > 200;

        if (!done)
        {
            reward = 1.0;
        }
        else if (steps_beyond_done == -1)
        {
            // Pole just fell!
            steps_beyond_done = 0;
            reward = 0;
        }
        else
        {
            if (steps_beyond_done == 0)
            {
                //AT_ASSERT(false); // Can't do this
            }
        }
        step_++;

        auto ret = af::array({4,1}, state);

        return {ret, reward, done};
    }

private:
    double gravity = 9.8;
    double masscart = 1.0;
    double masspole = 0.1;
    double total_mass = (masspole + masscart);
    double length = 0.5; // actually half the pole's length;
    double polemass_length = (masspole * length);
    double force_mag = 10.0;
    double tau = 0.02; // seconds between state updates;

    // Angle at which to fail the episode
    double theta_threshold_radians = 12 * 2 * M_PI / 360;
    double x_threshold = 2.4;
    int steps_beyond_done = -1;

    //fl::Variable state;
    float state[4];
    double reward;
    bool done;
    int step_ = 0;
};
