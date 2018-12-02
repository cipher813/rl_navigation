"""
Deep Q Network (DQN) helper file.
Project 1: Navigation
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
December 2018

Note that unityagents and gym are called only when specified in the run function.
"""
# from network import *
from agent import Vanilla, Double, PriorityReplay, Dueling, A3C, Rainbow

def train_gym(CHART_PATH, CHECKPOINT_PATH, agent_dict, module, timestamp, seed, score_target,
              n_episodes,max_t,e_start,e_end,e_decay): #agent_dict,
    """Trains OpenAI Gym environments."""
    import gym
    start = time.time()
    label = "gym"
    env = gym.make(module)
    ss = env.observation_space.shape[0]
    acts = env.action_space.n
    result_dict = {}
    for k,v in agent_dict.items():
        agent_name = k
        print(f"Agent: {k}")
        agent = v(ss, acts, seed)
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        e = e_start
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state,e)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward                                # update the score
                if done:                                       # exit loop if episode finished
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            e = max(e_end, e_decay*e) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=score_target:
                end = time.time()
                print(f'Environment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}\tRuntime: {(end-start)/60:.2f}')
                checkpath = CHECKPOINT_PATH + f'checkpoint-{timestamp}-{label}-{module}-{agent_name}.pth'
                torch.save(agent.qnetwork_local.state_dict(), checkpath)
                print(f"Checkpoint saved at {checkpath}")
                break
        result_dict[agent_name] = {
                        "scores":scores,
                        "clocktime":round((end-start)/60,2)
                        }
        pklpath = CHART_PATH + f"ResultDict-{timestamp}-{label}-{module}-{agent_name}.pkl"
        # pklpath = CHART_PATH + f"ResultDict-{timestamp}-{label}-{module}.pkl"
        with open(pklpath, 'wb') as handle:
            pickle.dump(result_dict, handle)
        print(f"Scores pickled at {pklpath}")
    return result_dict

def train_unity(PATH, CHART_PATH, CHECKPOINT_PATH, agent_dict, module, timestamp, seed, score_target,
                n_episodes,max_t,e_start,e_end,e_decay):
    """Trains Unity 3D Editor environments."""
    from unityagents import UnityEnvironment
    APP_PATH = PATH + f"data/{module}"
    label = "unity"
    start = time.time()
    env = UnityEnvironment(file_name=APP_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset()[brain_name]
    ss = len(env_info.vector_observations[0])
    acts = brain.vector_action_space_size
    result_dict = {}
    for k,v in agent_dict.items():
        agent_name = k
        print(f"Agent: {k}")
        # rainbow requires additional parameters
        if k=="Rainbow":
            agent = v(ss, acts, seed, num_atoms=51, Vmin=-10, Vmax=10)
        else:
            agent = v(ss,acts,seed)
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        e = e_start
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                state = env_info.vector_observations[0]  # get the current state
                action = agent.act(state,e)
                env_info = env.step(action)[brain_name]        # send the action to the environment
                next_state = env_info.vector_observations[0]   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]
                agent.step(state, action, reward, next_state, done)
                score += reward                                # update the score
                state = next_state                             # roll over the state to next time step
                if done:                                       # exit loop if episode finished
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            e = max(e_end, e_decay*e) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=score_target:
                end = time.time()
                print(f'Environment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}\tRuntime: {(end-start)/60:.2f}')
                checkpath = CHECKPOINT_PATH + f'checkpoint-{timestamp}-{label}-{agent_name}.pth'
                torch.save(agent.qnetwork_local.state_dict(), checkpath)
                print(f"Checkpoint saved at {checkpath}")
                break
        result_dict[agent_name] = {
                        "scores":scores,
                        "clocktime":round((end-start)/60,2)
                        }
        pklpath = CHART_PATH + f"ResultDict-{timestamp}-{label}-{agent_name}.pkl"
        with open(pklpath, 'wb') as handle:
            pickle.dump(result_dict, handle)
        print(f"Scores pickled at {pklpath}")
    return result_dict

def train_envs(PATH, CHART_PATH, CHECKPOINT_PATH, agent_dict, timestamp, env_dict, seed=0,
               n_episodes=3000,max_t=1000,e_start=0.4,e_end=0.01,e_decay=0.995):
    """Main trian function for all envs in env_dict."""
    rd = {}
    for k,v in env_dict.items():
        start = time.time()
        module = k
        platform = v[0]
        print(f"Begin training {module}-{platform}.")
        score_target = v[1]
        print(f"Module: {module}-{platform}")
        if platform == "gym":
            results = train_gym(CHART_PATH, CHECKPOINT_PATH, agent_dict, module, timestamp, seed, score_target,
                            n_episodes,max_t,e_start,e_end,e_decay)
        elif platform == "unity":
            results = train_unity(PATH, CHART_PATH, CHECKPOINT_PATH, agent_dict, module, timestamp, seed, score_target,
                          n_episodes,max_t,e_start,e_end,e_decay)
        else:
            print("Check your model and platform inputs.")
        rd[module] = results
        end = time.time()
        print(f"Finished training {module}-{platform} in {(end-start)/60:.2f} minutes.")
    pklpath = CHART_PATH + f"ResultDict-All-{timestamp}.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(rd, handle)
        print(f"Scores pickled at {pklpath}")
    return rd

def chart_results(CHART_PATH, pklfile):
    """Charts performance results by agent."""
    pklpath = CHART_PATH + pklfile
    timestamp = pklpath.split(".")[-2].split("-")[-1]

    with open(pklpath, 'rb') as handle:
        results = pickle.load(handle)
    for module in results.keys():
        mod_data = results[module]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for key in mod_data.keys():
            scores = mod_data[key]['scores']
            avg_scores = []
            for i in range(1,len(scores)+1):
                start = np.max(i-roll_length,0)
                end = i
                nm = np.sum(scores[start:end])
                dn = len(scores[start:end])
                avg_scores.append(nm/dn)
            plt.plot(np.arange(len(scores)), avg_scores,label=key)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.title(f"{module}")
            plt.legend()
        chartpath = CHART_PATH + f"NavigationTrainChart-{timestamp}-{module}-{key}.png"
        plt.savefig(chartpath)
        print(f"Chart saved at {chartpath}")
    plt.show()
    display(pd.DataFrame(results))
    return results
