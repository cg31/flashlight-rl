
#include <random>
#include <deque>
#include <vector>
#include <sstream>
#include <iostream>

#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"

#include "env/cartpole.h"

#include "gather.h"


float g_lr = 0.0005;
float g_gamma = 0.99;

int g_epochs = 5000;

int g_buffer_limit = 2000;
int g_batch_size = 32;

// Epsilon greedy exploration
float g_epsilon_start = 1.0;
float g_epsilon_final = 0.01;
float g_epsilon_decay = 500;


class ReplayBuffer
{
public:
    using Sample = std::tuple<af::array, int, float, af::array, float>;

    ReplayBuffer()
    {
    }

    void store(const af::array &s, int a, float r, const af::array &n, float d)
    {
        while (buffer.size() >= g_buffer_limit)
            buffer.pop_front();

        buffer.push_back({s, a, r, n, d});
    }

    auto sample(int num)
    {
        std::mt19937 engine(rd());
        std::uniform_int_distribution<> gen(0, buffer.size() - 1);

        std::vector<float> r_samples, d_samples;
        std::vector<int> a_samples;
        af::array as, an;

        std::set<int> seen;
        for (int i = 0; i < num; i++)
        {
            int idx = gen(engine);
            if (seen.find(idx) != seen.end())
                continue;

            seen.insert(idx);
            auto &s = buffer[idx];

            as = af::join(1, as, std::get<0>(s));
            an = af::join(1, an, std::get<3>(s));

            a_samples.push_back(std::get<1>(s));
            r_samples.push_back(std::get<2>(s));
            d_samples.push_back(std::get<4>(s));
        }

        auto s = fl::Variable(as, false);
        auto n = fl::Variable(an, false);

        auto a = af::array({1, a_samples.size()}, a_samples.data());
        auto r = fl::Variable(af::array({1, r_samples.size()}, r_samples.data()), false);
        auto d = fl::Variable(af::array({1, d_samples.size()}, d_samples.data()), false);

        return std::make_tuple(s, a, r, n, d);
    }

    size_t size()
    {
        return buffer.size();
    }

private:
    std::deque<Sample> buffer;
    std::random_device rd;
};


class DDQN
{
public:
    DDQN(int obs_dim, int act_dim, int hidden)
    {
        model.add(fl::Linear(obs_dim, hidden));
        model.add(fl::ReLU());
        model.add(fl::Linear(hidden, hidden));
        model.add(fl::ReLU());
        model.add(fl::Linear(hidden, act_dim));

        target.add(fl::Linear(obs_dim, hidden));
        target.add(fl::ReLU());
        target.add(fl::Linear(hidden, hidden));
        target.add(fl::ReLU());
        target.add(fl::Linear(hidden, act_dim));

        optimizer = std::make_unique<fl::AdamOptimizer>(model.params(), g_lr);
        buf = std::make_unique<ReplayBuffer>();

        sync();
    }

    int act(const af::array &state, float epsilon)
    {
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> random_float(0.0, 1.0);
        float coin = random_float(gen);

        if (coin > epsilon)
        {
            // greedy action
            auto out = model.forward(fl::Variable(state, false));
            af::array val, idx;
            af::max(val, idx, out.array(), 0);
            return idx.scalar<unsigned>();
        }
        else
        {
            // random action
            std::uniform_int_distribution<> random_int(0, 1);
            int act = random_int(gen);
            return act;
        }
    }

    void sync()
    {
        for (int i = 0; i < target.params().size(); i++)
        {
            target.setParams(model.param(i), i);
        }
    }

    void store(const af::array &state, int act, float rew, const af::array &next, float done)
    {
        buf->store(state, act, rew, next, done);
    }

    int buf_size()
    {
        return buf->size();
    }

    void update()
    {
        fl::MeanAbsoluteError criterion;

        for (int i=0; i < 10; i++)
        {
            auto [state, action, reward, next_state, done] = buf->sample(g_batch_size);

            auto q_values = model.forward(state);
            auto q_value = gather(q_values, action);

            auto q_values_target = target.forward(next_state);
            auto q_value_target = max(q_values_target, 0);

            auto q_value_expected = reward + g_gamma * q_value_target * done;
            auto loss = criterion.forward(q_value, q_value_expected);

            optimizer->zeroGrad();
            loss.backward();
            optimizer->step();
        }
        sync();
    }

    fl::Sequential model, target;
    std::unique_ptr<fl::AdamOptimizer> optimizer;
    std::unique_ptr<ReplayBuffer> buf;
    std::random_device rd;
};

int hidden = 128;
//int hidden = 8;

int main()
{
    fl::init();

    CartPoleEnv env;

    DDQN agent(4, 2, hidden);

    auto state = env.reset();
    float episode_reward = 0;

    for (int epoch = 0; epoch < g_epochs; epoch++)
    {
        //float epsilon = std::max(0.01, 0.08 - 0.01 * (epoch/200)); // Linear annealing from 8% to 1%
        float epsilon = g_epsilon_final + (g_epsilon_start - g_epsilon_final) * std::exp(-1.0 * epoch / g_epsilon_decay);

        auto action = agent.act(state, epsilon);
        auto [next_state, reward, done] = env.step(action);

        agent.store(state, action, reward/100.0, next_state, done ? 0 : 1);

        state = next_state;
        episode_reward += reward;

        if (done)
        {
            state = env.reset();
            std::cout << "episode: " << epoch << " | reward: " << episode_reward << std::endl;
            episode_reward = 0;
        }

        if (agent.buf_size() >= g_batch_size)
        {
            agent.update();
        }
    }

    return 0;
}
