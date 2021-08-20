
#include <random>
#include <vector>
#include <iostream>

#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"

#include "env/cartpole.h"

#include "gather.h"

const int epochs = 500;
const float g_gamma = 0.99;
const float g_lam = 0.97;
const int train_epoch_len = 1000;
const int obs_dim = 4;
const int act_dim = 2;
const int hidden_size = 64;
//const int hidden_size = 4;

float pi_lr = 3e-4;
float vf_lr = 1e-3;
int max_ep_len = 1000;
int local_steps_per_epoch = 4000;

int train_pi_iters = 80;
int train_v_iters = 80;
float target_kl = 0.01;
float clip_ratio = 0.2;


int multinomial(const fl::Variable &probs)
{
    std::random_device rd;
    std::mt19937 rnd_gen(rd());
    std::exponential_distribution<float> rng(1);

    std::vector<float> dist;
    int size = probs.elements();
    dist.resize(size);
    for (auto i=0; i < size; i++)
        dist[i] = rng(rnd_gen);

    auto q = af::array(probs.dims(), dist.data());
    auto s = probs.array() / q;

    af::array val, idx;
    af::max(val, idx, s, 0);

    return idx.scalar<unsigned>();
}


class Buffer
{
public:
    Buffer(int od, int ad, float g, float l) :
        obs_dim(od), act_dim(ad), gamma(g), lam(l)
    {
    }

    void store(const af::array &obs, int act, float rew, const af::array &val, const af::array &logp)
    {
        obs_buf.push_back(obs);
        act_buf.push_back(act);
        rew_buf.push_back(rew);
        val_buf.push_back(val);
        logp_buf.push_back(logp);
    }

    auto get()
    {
        int size = rew_buf.size();

        val_buf.push_back(af::constant(0.0f, 1));

        std::vector<af::array> adv_buf(size);
        auto last_gae_lam = af::constant(0.0f, 1);

        // advantage estimates with GAE-Lambda
        for (int i = size - 1; i >= 0; i--)
        {
            auto delta = rew_buf[i] + gamma * val_buf[i + 1] - val_buf[i];
            last_gae_lam = delta + gamma * lam * last_gae_lam;
            adv_buf[i] = last_gae_lam;
        }

        std::vector<af::array> ret_buf(size);
        last_gae_lam = af::constant(0.0f, 1);

        // advantage estimates with GAE-Lambda for ret
        for (int i = size - 1; i >= 0; i--)
        {
            last_gae_lam = rew_buf[i] + gamma * last_gae_lam;
            ret_buf[i] = last_gae_lam;
        }

        af::array obs, val, logp, adv, ret;

        for (int i=0; i < size; i++)
        {
            obs = af::join(1, obs, obs_buf[i]);
            val = af::join(1, val, val_buf[i]);
            logp = af::join(1, logp, logp_buf[i]);
            adv = af::join(1, adv, adv_buf[i]);
            ret = af::join(1, ret, ret_buf[i]);
        }

        auto act = af::array({1, size}, act_buf.data());
        auto mean = af::tile(af::mean(adv, {1}), adv.dims());
        auto stdtmp = af::stdev(adv, AF_VARIANCE_DEFAULT);
        auto stdev = af::tile(stdtmp, adv.dims());

        adv = (adv - mean) / stdev;

        obs_buf.clear();
        act_buf.clear();
        rew_buf.clear();
        val_buf.clear();
        logp_buf.clear();

        auto fobs = fl::Variable(obs, false);
        auto fact = fl::Variable(act, false);
        auto fret = fl::Variable(ret, false);
        auto fval = fl::Variable(val, false);
        auto flogp = fl::Variable(logp, false);
        auto fadv = fl::Variable(adv, false);

        return std::make_tuple(fobs, act, fret, fadv, flogp, fval);
    }

private:
    int obs_dim, act_dim;
    float gamma, lam;
    std::vector<af::array> obs_buf, val_buf, logp_buf;
    std::vector<int> act_buf;
    std::vector<float> rew_buf;
};


class PPO
{
public:
    PPO(int obs_dim, int act_dim, int hidden_size)
    {
        actor.add(fl::Linear(obs_dim, hidden_size));
        actor.add(fl::Tanh());
        actor.add(fl::Linear(hidden_size, hidden_size));
        actor.add(fl::Tanh());
        actor.add(fl::Linear(hidden_size, act_dim));
        actor.add(fl::LogSoftmax());

        critic.add(fl::Linear(obs_dim, hidden_size));
        critic.add(fl::ReLU());
        critic.add(fl::Linear(hidden_size, hidden_size));
        critic.add(fl::Tanh());
        critic.add(fl::Linear(hidden_size, 1));

        pi_optimizer = std::make_unique<fl::AdamOptimizer>(actor.params(), pi_lr);
        vf_optimizer = std::make_unique<fl::AdamOptimizer>(critic.params(), vf_lr);

        buf = std::make_unique<Buffer>(obs_dim, act_dim, g_gamma, g_lam);
    }

    auto policy(const fl::Variable &state, const af::array &act)
    {
        auto probs = actor.forward(state);
        auto logp = gather(probs, act);
        return logp;
    }

    auto value(const fl::Variable &state)
    {
        auto val = critic.forward(state);
        return val;
    }

    auto step(const fl::Variable &state)
    {
        auto probs = actor.forward(state);
        auto act = multinomial(probs);

        auto logp = probs.row(act);
        auto val = value(state);

        return std::make_tuple(act, val.array(), logp.array());
    }

    void store(const af::array &state, int act, float rew, const af::array &val, const af::array &logp)
    {
        buf->store(state, act, rew, val, logp);
    }

    auto compute_loss_pi(const fl::Variable &state, const af::array &act, const fl::Variable &logp_old, const fl::Variable &adv)
    {
        auto logp = policy(state, act);
        auto ratio = fl::exp(logp - logp_old);
        auto clip_adv = fl::clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv;
        auto loss_pi = fl::negate(fl::mean(fl::min(ratio * adv, clip_adv), {0,1}));
        return loss_pi;
    }

    auto compute_loss_v(const fl::Variable &state, const fl::Variable &ret)
    {
        auto val = value(state);
        auto diff = val - ret;
        auto loss_v = fl::mean(diff * diff, {0,1});
        return loss_v;
    }

    void update()
    {
        auto [state, act, ret, adv, logp, val] = buf->get();

        // train policy with multiple steps of gradient descent
        for (auto i=0; i < train_pi_iters; i++)
        {
            auto loss_pi = compute_loss_pi(state, act, logp, adv);

            pi_optimizer->zeroGrad();
            loss_pi.backward();
            pi_optimizer->step();
        }

        // value function learning
        for (auto i=0; i < train_v_iters; i++)
        {
            auto loss_v = compute_loss_v(state, ret);

            vf_optimizer->zeroGrad();
            loss_v.backward();
            vf_optimizer->step();
        }
    }

    fl::Sequential actor, critic;
    std::unique_ptr<fl::AdamOptimizer> pi_optimizer, vf_optimizer;
    std::unique_ptr<Buffer> buf;
};


int main()
{
    CartPoleEnv env;

    PPO agent(obs_dim, act_dim, hidden_size);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        auto state = env.reset();
        float ep_ret = 0;
        int ep_len = 0;

        std::vector<float> rets_log;
        std::vector<float> lens_log;

        for (int t = 0; t < local_steps_per_epoch; t++)
        {
            auto [action, value, logp] = agent.step(fl::Variable(state, false));
            auto [next_state, reward, done] = env.step(action);

            ep_ret += reward;
            ep_len += 1;

            agent.store(state, action, reward, value, logp);
            state = next_state;

            if (done)
            {
                rets_log.push_back(ep_ret);
                lens_log.push_back(ep_len);
                break;
            }
        }

        std::cout << "epoch: " << epoch
                  << " ret_mean: " << std::accumulate(rets_log.begin(), rets_log.end(), 0.) / rets_log.size()
                  << " len_mean: " << std::accumulate(lens_log.begin(), lens_log.end(), 0.) / lens_log.size()
                  << std::endl;

        agent.update();
    }

    return 0;
}

