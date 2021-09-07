from numpy.core.function_base import geomspace
# from ray.tune import analysis
from actor_critic import *

import ray
# from ray import tune
import json

from time import time

from numba import typed, typeof, njit, int64, types
from numba.experimental import jitclass
from numba.extending import overload_method, overload

@overload_method(types.misc.ClassInstanceType, 'clone')
def ol_bag_clone(inst,):
    if inst is GridWorld.class_type.instance_type:
        def impl(inst,):
            return GridWorld(obstacle=inst.obstacles)
        return impl
    elif inst is ActorCriticGridworld.class_type.instance_type:
        def impl(inst,):
            return ActorCriticGridworld(
                inst.env, inst.policy, alpha_actor=inst.alpha_actor,
                alpha_critic=inst.alpha_critic,
                lam=inst.lam
            )
        return impl
    elif inst is TabularSoftmaxPolicy.class_type.instance_type:
        def impl(inst,):
            return TabularSoftmaxPolicy(
                inst.env, eps=inst.eps
            )
        return impl
def serializer_gridworld(gw):
    return gw.obstacles

def deserializer_gridworld(obs):
    return GridWorld(obstacle=obs)

def serializer_policy(pol):
    return (pol.env.obstacles, pol.eps)

def deserializer_policy(obs, eps):
    return TabularSoftmaxPolicy(GridWorld(obstacle=obs), eps=eps)

def serializer_actorcritic(ac):
    return ac.env.obstacles, 

# @ray.remote
def objective(alpha_actor, alpha_critic, lam):
    seed=np.random.randint(42)
    # alpha_actor, alpha_critic, lam = config['alpha_actor'], config['alpha_critic'], config['lam']
    gw = get_gw_from_seed(seed)
    pol = TabularSoftmaxPolicy(gw,eps=0.0, seed=seed)
    reinforce = ActorCriticGridworld(
        gw, pol,
        alpha_actor=alpha_actor,
        alpha_critic=alpha_critic,
        lam=lam,
        seed=seed
        )
    eps = 1024
    a = time()
    for i in range(eps):
        gw.reset()
        reinforce.newepisode()
        while not gw.is_terminated():
            reinforce.act()
    reinforce.freeze()
    gw.reset()
    while not gw.is_terminated():
        reinforce.act()
    reward = gw.rw
    print(f"Time taken: {time()-a} seconds")
    print(f"Reward for alpha_actor:{alpha_actor} alpha_critic: {alpha_critic} lambda: {lam}={reward}")
    return {
        "alpha_actor": alpha_actor,
        "alpha_critic": alpha_critic,
        "lambda": lam,
        "reward": gw.rw
    }, reinforce.policy.theta

# analysis = tune.run(
#     objective,
#     config={
#         'alpha_actor': tune.loguniform(1e-6, 10),
#         'alpha_critic': tune.loguniform(1e-6, 10),
#         'lam':tune.loguniform(1e-6, 10),
#     }
# )

# print("Best config: ", analysis.get_best_config(
#     metric="avg_reward", mode="max"))

# print(analysis.result_df)
# print(analysis.best_result_df)

if __name__=="__main__":
    ray.init(local_mode=True)
    futures = [
        objective(i,j,k) 
        for i in np.geomspace(1e-7, 1, 10) 
        for j in np.geomspace(1e-7, 1, 10)
        for k in np.geomspace(1e-7, 1, 10)
    ]
    max_r = -np.inf
    best_pol= None
    for f, p in futures:
        if f['reward'] > max_r:
            max_r = f['reward']
            best_pol = p
    json.dump(futures, open("result.json", "w"))
    np.savez('best_policy', pol=best_pol)


