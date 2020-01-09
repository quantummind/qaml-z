import numpy as np
from scipy.optimize import basinhopping
from sklearn.metrics import accuracy_score

from contextlib import closing
from dwave_sapi2.util import qubo_to_ising, ising_to_qubo
from dwave_sapi2.fix_variables import fix_variables
from multiprocessing import Pool
import dwave_sapi2.remote
from dwave_sapi2.embedding import find_embedding,embed_problem,unembed_answer
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.core import solve_ising, await_completion
import os
import datetime
import time


a_time = 5
train_sizes = [100, 1000, 5000, 10000, 15000, 20000]
start_num = 9
end_num = 10

zoom_factor = 0.5
n_iterations = 8

flip_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))
flip_others_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))/2
flip_state = -1

AUGMENT_CUTOFF_PERCENTILE = 95
AUGMENT_SIZE = 7   # must be an odd number (since augmentation includes original value in middle)
AUGMENT_OFFSET = 0.0075

FIXING_VARIABLES = True

def hamiltonian_checker(s, C_i, C_ij, reg):
    if POS_WEIGHTS:
        return hamiltonian_orig_posweights(s, C_i, C_ij, reg)
    elif POS_NEG_WEIGHTS:
        return hamiltonian_orig_posnegweights(s, C_i, C_ij, reg)

def total_hamiltonian(s, C_i, C_ij):
    bits = len(s)
    h = 0 - np.dot(s, C_i)
    for i in range(bits):
        h += s[i] * np.dot(s[i+1:], C_ij[i][i+1:])
    return h
    
def hamiltonian(s, C_i, C_ij, mu, sigma, reg):
    s[np.where(s > 1)] = 1.0
    s[np.where(s < -1)] = -1.0
    bits = len(s)
    h = 0
    for i in range(bits):
        h += 2*s[i]*(-sigma[i]*C_i[i])
        for j in range(bits):
            if j > i:
                h += 2*s[i]*s[j]*sigma[i]*sigma[j]*C_ij[i][j]
            h += 2*s[i]*sigma[i]*C_ij[i][j] * mu[j]
    return h


def anneal(C_i, C_ij, mu, sigma, l, strength_scale, energy_fraction, ngauges, max_excited_states):
    url = "https://usci.qcc.isi.edu/sapi"
    token = "your-token"
    
    h = np.zeros(len(C_i))
    J = {}
    for i in range(len(C_i)):
        h_i = -2*sigma[i]*C_i[i]
        for j in range(len(C_ij[0])):
            if j > i:
                J[(i, j)] = 2*C_ij[i][j]*sigma[i]*sigma[j]
            h_i += 2*(sigma[i]*C_ij[i][j]*mu[j])
        h[i] = h_i

    vals = np.array(J.values())
    cutoff = np.percentile(vals, AUGMENT_CUTOFF_PERCENTILE)
    to_delete = []
    for k, v in J.items():
        if v < cutoff:
            to_delete.append(k)
    for k in to_delete:
        del J[k]

    isingpartial = []

    if FIXING_VARIABLES:
        Q, _  = ising_to_qubo(h, J)
        simple = fix_variables(Q, method='standard')
        new_Q = simple['new_Q']
        print('new length', len(new_Q))
        isingpartial = simple['fixed_variables']
    if (not FIXING_VARIABLES) or len(new_Q) > 0:
        cant_connect = True
        while cant_connect:
            try:
                print('about to call remote')
                conn = dwave_sapi2.remote.RemoteConnection(url, token)
                solver = conn.get_solver("DW2X")
                print('called remote', conn)
                cant_connect = False
            except IOError:
                print('Network error, trying again', datetime.datetime.now())
                time.sleep(10)
                cant_connect = True

        A = get_hardware_adjacency(solver)
        
        mapping = []
        offset = 0
        for i in range(len(C_i)):
            if i in isingpartial:
                mapping.append(None)
                offset += 1
            else:
                mapping.append(i - offset)
        if FIXING_VARIABLES:
            new_Q_mapped = {}
            for (first, second), val in new_Q.items():
                new_Q_mapped[(mapping[first], mapping[second])] = val
            h, J, _ = qubo_to_ising(new_Q_mapped)
        
        # run gauges
        nreads = 200
        qaresults = np.zeros((ngauges*nreads, len(h)))
        for g in range(ngauges):
            embedded = False
            for attempt in range(5):
                a = np.sign(np.random.rand(len(h)) - 0.5)
                h_gauge = h*a
                J_gauge = {}
                for i in range(len(h)):
                    for j in range(len(h)):
                        if (i, j) in J:
                            J_gauge[(i, j)] = J[(i, j)]*a[i]*a[j]
            
                embeddings = find_embedding(J.keys(), A)
                try:
                    (h0, j0, jc, new_emb) = embed_problem(h_gauge, J_gauge, embeddings, A, True, True)
                    embedded = True
                    break
                except ValueError:      # no embedding found
                    print('no embedding found')
                    embedded = False
                    continue
            
            if not embedded:
                continue
            
            # adjust chain strength
            rescale_couplers = strength_scale * max(np.amax(np.abs(np.array(h0))), np.amax(np.abs(np.array(list(j0.values())))))
    #         print('scaling by', rescale_couplers)
            for k, v in j0.items():
                j0[k] /= strength_scale
            for i in range(len(h0)):
                h0[i] /= strength_scale

            emb_j = j0.copy()
            emb_j.update(jc)
        
            print("Quantum annealing")
            try_again = True
            while try_again:
                try:
                    qaresult = solve_ising(solver, h0, emb_j, num_reads = nreads, annealing_time = a_time, answer_mode='raw')
                    try_again = False
                except:
                    print('runtime or ioerror, trying again')
                    time.sleep(10)
                    try_again = True
            print("Quantum done")

            qaresult = np.array(unembed_answer(qaresult["solutions"], new_emb, 'vote', h_gauge, J_gauge))
            qaresult = qaresult * a
            qaresults[g*nreads:(g+1)*nreads] = qaresult
        
        if FIXING_VARIABLES:
            j = 0
            for i in range(len(C_i)):
                if i in isingpartial:
                    full_strings[:, i] = 2*isingpartial[i] - 1
                else:
                    full_strings[:, i] = qaresults[:, j]
                    j += 1
        else:
            full_strings = qaresults
        
        s = full_strings
        energies = np.zeros(len(qaresults))
        s[np.where(s > 1)] = 1.0
        s[np.where(s < -1)] = -1.0
        bits = len(s[0])
        for i in range(bits):
            energies += 2*s[:, i]*(-sigma[i]*C_i[i])
            for j in range(bits):
                if j > i:
                    energies += 2*s[:, i]*s[:, j]*sigma[i]*sigma[j]*C_ij[i][j]
                energies += 2*s[:, i]*sigma[i]*C_ij[i][j] * mu[j]
        
        unique_energies, unique_indices = np.unique(energies, return_index=True)
        ground_energy = np.amin(unique_energies)
#         print('ground energy', ground_energy)
        if ground_energy < 0:
            threshold_energy = (1 - energy_fraction) * ground_energy
        else:
            threshold_energy = (1 + energy_fraction) * ground_energy
        lowest = np.where(unique_energies < threshold_energy)
        unique_indices = unique_indices[lowest]
        if len(unique_indices) > max_excited_states:
            sorted_indices = np.argsort(energies[unique_indices])[-max_excited_states:]
            unique_indices = unique_indices[sorted_indices]
        final_answers = full_strings[unique_indices]
        print('number of selected excited states', len(final_answers))
        
        return final_answers
        
    else:
        final_answer = []
        for i in range(len(C_i)):
            if i in isingpartial:
                final_answer.append(2*isingpartial[i] - 1)
        final_answer = np.array(final_answer)
        return np.array([final_answer])

def create_data(sig, bkg):
    n_classifiers = sig.shape[1]
#     predictions = np.concatenate((sig, bkg))
    predictions = np.concatenate((np.sign(sig), np.sign(bkg)))
    predictions = np.transpose(predictions) / float(n_classifiers)
    y = np.concatenate((np.ones(len(sig)), -np.ones(len(bkg))))
    return predictions, y

def create_augmented_data(sig, bkg):
    offset = AUGMENT_OFFSET
    scale = AUGMENT_SIZE

    n_samples = len(sig) + len(bkg)
    n_classifiers = sig.shape[1]
    predictions_raw = np.concatenate((sig, bkg))
    predictions_raw = np.transpose(predictions_raw)
    predictions = np.zeros((n_classifiers * scale, n_samples))
    for i in range(n_classifiers):
        for j in range(scale):
            predictions[i*scale + j] = np.sign(predictions_raw[i] + (j-scale//2)*offset) / (n_classifiers * scale)
    y = np.concatenate((np.ones(len(sig)), -np.ones(len(bkg))))
    print('predictions', predictions)
    return predictions, y

def ensemble(predictions, weights):
    ensemble_predictions = np.zeros(len(predictions[0]))
    
    if POS_NEG_WEIGHTS:
        return np.sign(np.dot(predictions.T, weights))
    else:
        return np.sign(np.dot(predictions.T, weights/2 + 0.5)/n_classifiers)
print('loading data')
sig = np.loadtxt('sig.csv')
bkg = np.loadtxt('bkg.csv')
sig_pct = float(len(sig)) / (len(sig) + len(bkg))
bkg_pct = float(len(bkg)) / (len(sig) + len(bkg))
print('loaded data')

n_folds = 10
num = 0

for train_size in train_sizes:
    print('training with size', train_size)
    sig_indices = np.arange(len(sig))
    bkg_indices = np.arange(len(bkg))
    
    remaining_sig = sig_indices
    remaining_bkg = bkg_indices
    fold_generator = np.random.RandomState(0)
    for f in range(n_folds):
        if num >= end_num:
            break
        print('fold', f)
        train_sig = fold_generator.choice(remaining_sig, size=int(train_size*sig_pct), replace=False)
        train_bkg = fold_generator.choice(remaining_bkg, size=int(train_size*bkg_pct), replace=False)
        
        remaining_sig = np.delete(remaining_sig, train_sig)
        remaining_bkg = np.delete(remaining_bkg, train_bkg)
        
        test_sig = np.delete(sig_indices, train_sig)
        test_bkg = np.delete(bkg_indices, train_bkg)

        if AUGMENT:
            predictions_train, y_train = create_augmented_data(sig[train_sig], bkg[train_bkg])
            predictions_test, y_test = create_augmented_data(sig[test_sig], bkg[test_bkg])
        else:
            predictions_train, y_train = create_data(sig[train_sig], bkg[train_bkg])
            predictions_test, y_test = create_data(sig[test_sig], bkg[test_bkg])
        print('split data')
        
        if num < start_num:
            num += 1
            continue

        # create C_ij and C_i matrices
        n_classifiers = len(predictions_train)
        C_ij = np.zeros((n_classifiers, n_classifiers))
        C_i = np.dot(predictions_train, y_train)
        for i in range(n_classifiers):
            for j in range(n_classifiers):
                C_ij[i][j] = np.dot(predictions_train[i], predictions_train[j])

        print('created C_ij and C_i matrices')


        mu0 = np.zeros(n_classifiers)
        sigma0 = np.ones(n_classifiers)
        mu = np.copy(mu0)
        sigma = np.copy(sigma0)
        reg = 0.0
        l0 = reg*np.amax(np.diagonal(C_ij)*sigma*sigma - 2*sigma*C_i)
        strengths = [3.0, 1.0, 0.5, 0.2] + [0.1]*(n_iterations - 4)
        energy_fractions = [0.08, 0.04, 0.02] + [0.01]*(n_iterations - 3)
        gauges = [50, 10] + [1]*(n_iterations - 2)
        max_states = [16, 4] + [1]*(n_iterations - 2)     # cap the number of excited states accepted per iteration

        if UPDATING_HAMILTONIAN:
            mus = [np.zeros(n_classifiers)]
            iterations = n_iterations
            for i in range(iterations):
                print('iteration', i)
                l = reg*np.amax(np.diagonal(C_ij)*sigma*sigma - 2*sigma*C_i)
                new_mus = []
                for mu in mus:
                    excited_states = anneal(C_i, C_ij, mu, sigma, l, strengths[i], energy_fractions[i], gauges[i], max_states[i])
                    for s in excited_states:
                        new_energy = total_hamiltonian(mu + s*sigma*zoom_factor, C_i, C_ij) / (train_size - 1)
                        flips = np.ones(len(s))
                        for a in range(len(s)):
                            temp_s = np.copy(s)
                            temp_s[a] = 0
                            old_energy = total_hamiltonian(mu + temp_s*sigma*zoom_factor, C_i, C_ij) / (train_size - 1)
                            energy_diff = new_energy - old_energy
                            if energy_diff > 0:
                                flip_prob = flip_probs[i]
                                flip = np.random.choice([1, flip_state], size=1, p=[1-flip_prob, flip_prob])[0]
                                flips[a] = flip
                            else:
                                flip_prob = flip_others_probs[i]
                                flip = np.random.choice([1, flip_state], size=1, p=[1-flip_prob, flip_prob])[0]
                                flips[a] = flip
                        flipped_s = s * flips
                        new_mus.append(mu + flipped_s*sigma*zoom_factor)
                sigma *= zoom_factor
                mus = new_mus
                
                np.save('./mus' + str(try_number) + '/mus' + str(train_size) + 'fold' + str(f) + 'iter' + str(i) + '.npy', np.array(mus))
            for mu in mus:
                print('final accuracy on train set', accuracy_score(y_train, ensemble(predictions_train, mu)))
                print('final accuracy on test set', accuracy_score(y_test, ensemble(predictions_test, mu)))
        num += 1