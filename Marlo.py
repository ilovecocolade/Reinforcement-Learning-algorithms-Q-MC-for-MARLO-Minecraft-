#!/usr/bin/env python
# $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
from collections import defaultdict
import marlo
from Agent import Agent
from GameState import GameState
import numpy as np
from bs4 import BeautifulSoup
mission_name = 'MarLo-FindTheGoal-v0'
from marlo.base_env_builder import dotdict
from MonteCarlo import MonteCarlo
from Q import Q

# config
client_pool = [('127.0.0.1', 10000)]
join_tokens = marlo.make(mission_name, params={ "client_pool": client_pool,
                                               "suppress_info":False,
                                               "skip_steps":2,"step_sleep": 0.5})

# As this is a single agent scenario,
# there will just be a single token
assert len(join_tokens) == 1
join_token = join_tokens[0]

# define the environment variable and reset observation
env = marlo.init(join_token)

env.default_base_params.allowDiscreteMovement=True
env.default_base_params.allowContinuousMovement=False
env.default_base_params.allowAbsoluteMovement=False
env.mission_spec.removeAllCommandHandlers()
params=dotdict({"allowDiscreteMovement": True,"allowContinuousMovement": False,"allowAbsoluteMovement": False,"skip_steps":3,"step_sleep": 0.5,"tick_length":5,
                "prioritise_offscreen_rendering":True})
env.setup_action_commands(params)


#observation = env.reset()


# grab map information
mission_xml =  BeautifulSoup(env.params['mission_xml'], features="xml")
map_spec = mission_xml.find('specification')
placement = mission_xml.find('Placement')
map_dimension = [int(map_spec.contents[1].text), int(map_spec.contents[2].text),int(map_spec.contents[3].text)]
mission_available_moves = env.params['comp_all_commands']

num_episodes = 300
gamma = [1, .6, .3]
alpha = [1, .6, .3]
max_simulation_time = 120

# Input learning method
# MC - monte carlo, Q - Q learning
algorithm = 'Q'

for g in gamma:
    for a in alpha:
        if algorithm =='MC':
            # instantiate an Agent object
            mc = MonteCarlo(mission_name, env,num_episodes, g, max_simulation_time,a )
            mc.mc_prediction(filename='',iteration_number=0)
        elif algorithm =='Q':
            # instantiate an Agent object
            q = Q(mission_name, env ,num_episodes, g, a, max_simulation_time  )
            q.q_prediction()