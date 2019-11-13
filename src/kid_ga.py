import torch
import numpy as np


def run_kid_run(start_state, action_sequence, model, clip_val):
    action_sequence = clip_actions(action_sequence, clip_val)
    p_states = list()
    r_states = list()
    s_states = list()
    p_states.append(start_state['p_i'])
    r_states.append(start_state['r_i'])
    s_states.append(start_state['s_i'])

    for i in range(len(action_sequence)):
        p_next = model.forward_kinematics_model.forward(p_states[i], action_sequence[i])
        r_next = model.perceptual_model.decode(p_next)
        s_next = model.sensory_model.decode(r_next)
        p_states.append(p_next)
        r_states.append(r_next)
        s_states.append(s_next)

    return s_states, r_states, p_states


def calc_loss(target_state, state_sequence, action_sequence):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    mse = torch.nn.MSELoss()
    target_loss = 0
    control_loss = 0
    for i in range(len(action_sequence)):
        control_loss = control_loss + mse(torch.zeros(1, 3, device=device), action_sequence[i])
        # target_loss = target_loss + mse(target_state, state_sequence[-1])
    loss = target_loss + 0.1 * control_loss / len(action_sequence) + 10 * mse(target_state, state_sequence[-1])
    return loss


def clip_actions(actions, clip_val):
    for i in range(len(actions)):
        action = actions[i]
        r = torch.norm(action).detach()
        if r > clip_val:
            action = action / r * clip_val
            actions[i] = action
    return actions


def gen_actions(**kwargs):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if 'N' in kwargs:
        N = kwargs['N']
        actions = list()
        for i in range(N):
            action = torch.randn(1, 3, requires_grad=True, device=device)
            actions.append(action)
    elif 'actions' in kwargs:
        parent = kwargs['actions']
        actions = list()
        noise_lvl = kwargs['noise_lvl']
        for i in range(len(parent)):
            action = parent[i] + torch.randn(1, 3, requires_grad=True, device=device) * noise_lvl
            action = torch.tensor(action)
            actions.append(action)
    return actions


def calc_fitness(population, start_state, target_state, model, clip_val):
    fitness = np.zeros(len(population))
    for i in range(len(population)):
        action_sequence = population[i]
        s_states, r_states, p_states = run_kid_run(start_state, action_sequence, model, clip_val)
        state_sequence = r_states
        fitness[i] = calc_loss(target_state['r_i'], state_sequence, action_sequence)
    return fitness


def seed_population(N, pop_size):
    population = list()
    for i in range(pop_size):
        population.append(gen_actions(N=N))
    return population


def optimize_seq(actions, start, target, model, steps, learning_rate, clip_val):
    for i in range(len(actions)):
        actions[i] = torch.tensor(actions[i])
    optimizer = torch.optim.Adam(actions, lr=learning_rate)
    for i in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        s_states, r_states, p_states = run_kid_run(start, actions, model, clip_val)
        loss = calc_loss(target['r_i'], r_states, actions)
        loss.backward(retain_graph=True)
        optimizer.step()
        # print(i, loss, end='\r')
    for i in range(len(actions)):
        actions[i] = torch.tensor(actions[i])
    return actions


def optimize_pop(population, start, target, model, steps, learning_rate, clip_val):
    for i in range(len(population)):
        population[i] = optimize_seq(population[i], start, target, model, steps, learning_rate, clip_val)
    return population


def select_sequences(population, fitness, n_cut, n_children, noise_lvl):
    # print(np.min(fitness))
    indices = np.argsort(fitness)
    s_population = list()
    s_fitness = list()
    for i in range(len(indices)):
        s_population.append(population[indices[i]])
        s_fitness.append(fitness[indices[i]])
    s_population = s_population[0:n_cut]
    s_fitness = s_fitness[0:n_cut]
    population = s_population
    for i in range(len(s_population)):
        for j in range(n_children):
            population.append(gen_actions(actions=s_population[i], noise_lvl=noise_lvl))
    return population


def evolve(population, start, target, model, clip_val, genetic_args):
    if genetic_args['optimize']:
        steps = genetic_args['steps']
        learning_rate = genetic_args['learning_rate']
        population = optimize_pop(population, start, target, model, steps, learning_rate, clip_val)
    fitness = calc_fitness(population, start, target, model, clip_val)
    n_cut = genetic_args['n_cut']
    n_children = genetic_args['n_children']
    noise_lvl = genetic_args['noise_lvl']
    population = select_sequences(population, fitness, n_cut, n_children, noise_lvl)
    return population


def get_sequence(start, target, model, clip_val, genetic_args):
    population = seed_population(genetic_args['seq_length'], genetic_args['pop_size'])
    for i in range(genetic_args['generations']):
        population = evolve(population, start, target, model, clip_val, genetic_args)
    actions = population[0]
    s_states, r_states, p_states = run_kid_run(start, actions, model, clip_val)
    return s_states, r_states, p_states


def gen_rand_seq(dataset, model, clip_val, genetic_args):
    import numpy as np
    start_idx = np.random.choice(int(dataset.num_points) - 2) + 1
    target_idx = np.random.choice(int(dataset.num_points) - 2) + 1
    start = dataset.get_samples([start_idx])
    target = dataset.get_samples([target_idx])
    start = model.forward(**start)
    target = model.forward(**target)
    s_states, r_states, p_states = get_sequence(start, target, model, clip_val, genetic_args)
    return s_states, r_states, p_states, target


def build_dataset(num_seqs, dataset, model, clip_val, genetic_args):
    import time
    r_target = list()
    p_t = list()
    p_next = list()
    for i in range(num_seqs):
        t = time.time()
        s_states, r_states, p_states, target = gen_rand_seq(dataset, model, clip_val, genetic_args)
        for j in range(len(s_states) - 1):
            r_target.append(torch.tensor(target['r_i']).cpu())
            p_t.append(torch.tensor(p_states[j]).cpu())
            p_next.append(torch.tensor(p_states[j + 1]).cpu())
        print(i, time.time() - t, ' seconds')
    return r_target, p_t, p_next