import numpy as np
from sklearn.metrics import accuracy_score
from scipy.integrate import simps
from scipy.interpolate import interp1d


AUGMENT_SIZE = 7   # must be an odd number (since augmentation includes original value in middle)
AUGMENT_OFFSET = 0.0075
annnealing_time = 400

POISSON = True
b_start = 1000000
b_end = 2000000

def create_data(sig, bkg):
    n_classifiers = sig.shape[1]
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
    return predictions, y

def ensemble(predictions, weights):
    n_classifiers = len(weights)
    return np.dot(predictions.T, weights)

def auc(predictions, y_test, test_sig, test_bkg, poisson):
    cutoff = -1.0
    increments = 1000
    bkg_rejection = [0.0]
    sig_efficiency = [1.0]
    predictions /= np.amax(np.abs(predictions))
    
    for j in range(increments+1):
        cutoff += j * (2.0/increments)
        cut_predictions = np.sign(predictions - cutoff)
        agree = (y_test == cut_predictions).astype(np.float64)
        agree *= poisson
        se = np.sum(agree[np.where(y_test == 1.0)]) / len(test_sig)
        br = np.sum(agree[np.where(y_test == -1.0)]) / len(test_bkg)
        if br > 0 and se < sig_efficiency[-1] and br > bkg_rejection[-1] and se > 0:
            sig_efficiency.append(se)
            bkg_rejection.append(br)
        elif se == 0.0:
            break
    
    # add far term to make up for poisson
    bkg_rejection.append(2.0)
    sig_efficiency.append(0.0)
    
    bkg_rejection = np.array(bkg_rejection)
    sig_efficiency = np.array(sig_efficiency)
    
    sort = np.argsort(bkg_rejection)
    bkg_rejection = bkg_rejection[sort]
    sig_efficiency = sig_efficiency[sort]
    
    return bkg_rejection, sig_efficiency

sig = np.loadtxt('sig.csv')
bkg = np.loadtxt('bkg.csv')
sig_pct = float(len(sig)) / (len(sig) + len(bkg))
bkg_pct = float(len(bkg)) / (len(sig) + len(bkg))
print('loaded data')

train_sizes = [100, 1000, 5000, 10000, 15000, 20000]
n_folds = 10

if POISSON:
    poisson_runs = 5
else:
    poisson_runs = 1
y_score_count = 0
for i in range(len(train_sizes)):
    train_size = train_sizes[i]
    print('training with size', train_size)
    sig_indices = np.arange(len(sig))
    bkg_indices = np.arange(len(bkg))
    
    remaining_sig = sig_indices
    remaining_bkg = bkg_indices
    
    cts = np.zeros(n_folds*poisson_runs)
    
    fold_generator = np.random.RandomState(0)
    valid_generator = np.random.RandomState(0)
    for f in range(n_folds):
        print('fold', f)
        train_sig = fold_generator.choice(remaining_sig, size=int(train_size*sig_pct), replace=False)
        train_bkg = fold_generator.choice(remaining_bkg, size=int(train_size*bkg_pct), replace=False)
        
        remaining_sig = np.delete(remaining_sig, train_sig)
        remaining_bkg = np.delete(remaining_bkg, train_bkg)
        
        test_sig = np.delete(sig_indices, train_sig)
        test_bkg = np.delete(bkg_indices, train_bkg)
        
        valid_sig = valid_generator.choice(test_sig, size=int(train_size*sig_pct))
        valid_bkg = valid_generator.choice(test_bkg, size=int(train_size*bkg_pct))
        
        test_sig = np.delete(test_sig, valid_sig)
        test_bkg = np.delete(test_bkg, valid_bkg)

        predictions_test, y_test = create_augmented_data(sig[test_sig], bkg[test_bkg])
        predictions_valid, y_valid = create_augmented_data(sig[valid_sig], bkg[valid_bkg])
        excited_weights = np.load('./mus/mus' + str(train_size) + 'fold' + str(f) + '.npy')
        
        for p in range(poisson_runs):
            if POISSON:
                poisson = np.random.poisson(1.0, len(y_test))
                valid_poisson = np.random.poisson(1.0, len(y_valid))
            else:
                poisson = np.ones(len(y_test))
            
            excited_predictions = []
            bkg_grid = np.linspace(0, 1.05, num=1000)
            sig_efficiencies = np.zeros((len(excited_weights), len(bkg_grid)))
            valid_sig_efficiencies = np.zeros((len(excited_weights), len(bkg_grid)))
            for w in range(len(excited_weights)):
                continuous_predictions = ensemble(predictions_test, excited_weights[w])
                bkg_rejection, sig_efficiency = auc(continuous_predictions, y_test, test_sig, test_bkg, poisson)
                interp = interp1d(bkg_rejection, sig_efficiency, kind='cubic')
                
                valid_continuous_predictions = ensemble(predictions_valid, excited_weights[w])
                valid_bkg_rejection, valid_sig_efficiency = auc(valid_continuous_predictions, y_valid, valid_sig, valid_bkg, valid_poisson)
                valid_interp = interp1d(valid_bkg_rejection, valid_sig_efficiency, kind='cubic')
                
                sig_efficiencies[w] = interp(bkg_grid)
                valid_sig_efficiencies[w] = valid_interp(bkg_grid)
                for b in range(len(bkg_grid)):
                    if bkg_grid[b] > bkg_rejection[-2]:
                        sig_efficiencies[w][b] = 0
                    if bkg_grid[b] > valid_bkg_rejection[-2]:
                        valid_sig_efficiencies[w][b] = 0
            
            # take supremum among signal efficiency curves
            sig_efficiency_ind = np.argmax(valid_sig_efficiencies, axis=0)
            sig_efficiency_max = sig_efficiencies[(sig_efficiency_ind, np.arange(len(bkg_grid)))]
            continuous_auc = simps(sig_efficiency_max, bkg_grid)
            print('fold auc', continuous_auc)
            
            cts[f*poisson_runs + p] = continuous_auc
    
        
    print('auroc mean', np.mean(cts))
    print('auroc stdev', np.std(cts))