import math
import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import flags
from matplotlib import pyplot as plt

_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_RAND = 1000
_MOVE_MIDDLE = 2000
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

actions_list = [
    _NO_OP,
    _SELECT_ARMY,
    _SELECT_POINT,
    _MOVE_SCREEN,
    _MOVE_RAND,
    _MOVE_MIDDLE
]


def epsilon_greedy(steps_done):
    return 0.025 + (0.9 - 0.025) * math.exp(-1. * steps_done / 2500)

# define the state
def get_state(obs):
    # get the positions of the marine and the beacon
    ai_view = obs.observation['screen'][_AI_RELATIVE]
    ai_selected = obs.observation['screen'][_AI_SELECTED]
    beaconxs, beaconys = (ai_view == _AI_NEUTRAL).nonzero()
    marinexs, marineys = (ai_view == _AI_SELF).nonzero()
    marinex, mariney = marinexs.mean(), marineys.mean()
        
    marine_on_beacon = np.min(beaconxs) <= marinex <=  np.max(beaconxs) and np.min(beaconys) <= mariney <=  np.max(beaconys)
    # a=1 marine is selected
    # a=0 marine is not selected
    marine_selected = int((ai_selected == 1).any())
    int_m_beacon = int(marine_on_beacon)
    
    return (marine_selected, int_m_beacon), [beaconxs, beaconys]

class Agent3(base_agent.BaseAgent):
    def __init__(self, load_qt=None):
        super(Agent3, self).__init__()
        self.qtable = QTable(actions_list, load_qt=load_qt)
        
    def step(self, obs):
        '''Step function gets called automatically by pysc2 environment'''
        super(Agent3, self).step(obs)
        state, beacon_pos = get_state(obs)
        action = self.qtable.get_action(state)
        func = actions.FunctionCall(_NO_OP, [])
        
        if actions_list[action] == _NO_OP:
            func = actions.FunctionCall(_NO_OP, [])
        elif state[0] and actions_list[action] == _MOVE_SCREEN:
            beacon_x, beacon_y = beacon_pos[0].mean(), beacon_pos[1].mean()
            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon_y, beacon_x]])
        elif actions_list[action] == _SELECT_ARMY:
            func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        elif state[0] and actions_list[action] == _SELECT_POINT:
            ai_view = obs.observation['screen'][_AI_RELATIVE]
            backgroundxs, backgroundys = (ai_view == _BACKGROUND).nonzero()
            point = np.random.randint(0, len(backgroundxs))
            backgroundx, backgroundy = backgroundxs[point], backgroundys[point]
            func = actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [backgroundy, backgroundx]])
        elif state[0] and actions_list[action] == _MOVE_RAND:
            # move randomly
            beacon_x, beacon_y = beacon_pos[0].max(), beacon_pos[1].max()
            movex, movey = np.random.randint(beacon_x, 64), np.random.randint(beacon_y, 64)
            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [movey, movex]])
        elif state[0] and actions_list[action] == _MOVE_MIDDLE:
            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [32, 32]])
        return state, action, func

class QTable(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, load_qt=None):
        self.learning_rate = learning_rate
        self.actions = actions
        self.reward_decay = reward_decay
        self.states_list = set()
        self.load_qt = load_qt

        self.q_table = np.zeros((0, len(actions_list))) # create a Q table
        
    def get_action(self, state):
        if not self.load_qt and np.random.rand() < epsilon_greedy(steps):
            return np.random.randint(0, len(self.actions))
        else:
            if state not in self.states_list:
		self.q_table = np.vstack([self.q_table, np.zeros((1, len(actions_list)))])
		self.states_list.add(state)
            idx = list(self.states_list).index(state)
            q_values = self.q_table[idx]
            return int(np.argmax(q_values))

    def update_qtable(self, state, next_state, action, reward):
        if state not in self.states_list:
            self.q_table = np.vstack([self.q_table, np.zeros((1, len(actions_list)))])
	    self.states_list.add(state)
        if next_state not in self.states_list:
	    self.q_table = np.vstack([self.q_table, np.zeros((1, len(actions_list)))])
	    self.states_list.add(next_state)
        # how much reward 
        state_idx = list(self.states_list).index(state)
        next_state_idx = list(self.states_list).index(next_state)
        # calculate q labels
        q_state = self.q_table[state_idx, action]
        q_next_state = self.q_table[next_state_idx].max()
        q_targets = reward + (self.reward_decay * q_next_state)
        # calculate our loss 
        loss = q_targets - q_state
        # update the q value for this state/action pair
        self.q_table[state_idx, action] += self.learning_rate * loss
        return loss
    

FLAGS = flags.FLAGS
FLAGS(['run_sc2'])
viz = False
save_replay = False
steps_per_episode = 0 # 0 actually means unlimited
MAX_EPISODES =35
MAX_STEPS = 400
steps = 0
all_losses = []
all_nums = []
all = []
# create a map
beacon_map = maps.get('MoveToBeacon')

# create an envirnoment
with sc2_env.SC2Env(agent_race=None,
                    bot_race=None,
                    difficulty=None,
                    map_name=beacon_map,
                    visualize=viz) as env:
    agent = Agent3()
    for i in range(MAX_EPISODES):
        print('Starting episode {}'.format(i))
        all_nums.append(i)
        ep_reward = 0
        obs = env.reset()
        for j in range(MAX_STEPS):
            steps += 1
            state, action, func = agent.step(obs[0])
            obs = env.step(actions=[func])
            next_state, _ = get_state(obs[0])
            reward = obs[0].reward
            ep_reward += reward
            loss = agent.qtable.update_qtable(state, next_state, action, reward)
            all_losses.append(loss)
        print('Episode Reward: {}, Explore threshold: {}, Q loss: {}'.format(ep_reward, get_eps_threshold(steps), loss))
        all.append(loss)
    if save_replay:
        env.save_replay(Agent3.__name__)

arr_x = np.array(all_nums)
arr_y = np.array(all_losses)
array = np.arange(14000)
all_a = np.array(all)
med = np.arange(all_a.shape)
print(array.shape)
print(arr_x.shape)
print(arr_y.shape)
plt.title("Q-Loss changes")
plt.xlabel("episodes")
plt.ylabel("Q Loss")
plt.plot(med, all_a)
plt.savefig('q.png')
print('(marine_sel, marine_beac)', '[do nothing, select marine, deselect marine, move beacon, move random, move middle] ')
for state in agent.qtable.states_list:
    print(state, agent.qtable.q_table[list(agent.qtable.states_list).index(state)])


