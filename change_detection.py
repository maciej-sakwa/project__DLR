import numpy as np

'''
ICI-based univariate change detection test. 
'''


class ChangeDetectionTest():
    def __init__(self, fit_window:int, sequence_window:int, gamma = 1):
        self.fit_window = fit_window
        self.sequence_window = sequence_window
        self._n_sequences = self.fit_window // self.sequence_window

        self.gamma = gamma

        # Placeholders (TODO: are they needed??)
        self.CI_M = None
        self.CI_V = None

        self.means = None
        self.vars = None

        self.h_coeff = None

    def __repr__(self) -> str:
        rep = f'CDT params: \n' + \
        f'- fitting window length: {self.fit_window} \n' + \
        f'- sequence window lenght: {self.sequence_window} \n' + \
        f'- number of seqences: {self._n_sequences}'

        return rep

    def __call__(self, new_data) -> bool:
        """Designed to be used in a loop. Insert new data sequence to determine 
           if a change is detected in the *previous* sequence.

        Args:
            new_data (list or array): New sequence of data of the length equal to declared sequence window.

        Returns:
            bool: True if no change is detected, False otherwise.
        """

        assert len(new_data) == self.sequence_window, 'The incoming data length has to be the same as the declared sequence window'
        
        # Calculate features for new data
        mean_new_data = np.mean(new_data)
        vars_new_data = np.var(new_data)

        # Append the features to existing arrays
        self.means = np.append(self.means, mean_new_data)
        self.vars = np.append(self.vars, (vars_new_data / (self.sequence_window - 1)) ** self.h_coeff)

        # Find new CIs
        CI_M_new = self._get_ci(self.means)
        CI_V_new = self._get_ci(self.vars)
        
        # Checks
        self.CI_M = self._check_confidence_intervals(self.CI_M, CI_M_new)
        self.CI_V = self._check_confidence_intervals(self.CI_V, CI_V_new)

        if (self.CI_M is None) or (self.CI_V is None):
            return False

        return True


    def fit(self, data) -> None:
        """Fit the ICI-based change detection on the initial data string.

        Args:
            data (list or array): Initial data string of the lenght equal to the defined fitting window.
        """

        assert len(data) == self.fit_window, 'The fit data length has to be the same as the declared fitting window'

        # Convert to list          
        setup_window = [data[i] for i in range(len(data))]

        # Create lists of seqence means and vars from setup sequences
        means_setup = [np.mean(setup_window[i*self.sequence_window:(i+1)*self.sequence_window]) for i in range(self._n_sequences)]
        vars_setup = [np.var(setup_window[i*self.sequence_window:(i+1)*self.sequence_window]) for i in range(self._n_sequences)]

        # Normalize the vars list export h_coeff
        self.h_coeff = 1 - (np.mean(setup_window) * self._skewness(setup_window)) / 3 * np.var(setup_window)**2
        vars_setup = (np.array(vars_setup) / (self.sequence_window - 1)) ** self.h_coeff   

        # Put both features as arrays
        self.means = np.array(means_setup)
        if isinstance(vars_setup, np.ndarray):
            self.vars = vars_setup
        else: 
            self.vars = np.array(vars_setup)

        self.CI_M = self._get_ci(self.means)
        self.CI_V = self._get_ci(self.vars)

    def _confidcence_interval(self, mu: float, sigma:float) -> tuple:
        ci_min = mu - self.gamma * sigma
        ci_max = mu + self.gamma * sigma

        # Equal has to be allowed for interpolation based fills
        assert ci_min <= ci_max, f'ci_min should be lower then ci_max, while {ci_min} is higher then {ci_max}'
        return (ci_min, ci_max)

    def _get_ci_params(self, data:np.ndarray) -> tuple:
        mu = np.mean(data)
        sigma = np.std(data) / np.sqrt(self._n_sequences)
        return mu, sigma

    def _get_ci(self, data:np.ndarray) -> tuple:
        mu, sigma = self._get_ci_params(data)
        return self._confidcence_interval(mu, sigma)
    
    def _check_confidence_intervals(self, CI_1: tuple, CI_2: tuple):
        min1, max1 = CI_1
        min2, max2 = CI_2
        
        # Find the maximum of the minimums and the minimum of the maximums
        common_min = max(min1, min2)
        common_max = min(max1, max2)
        
        # Check if there is an overlap
        if common_min <= common_max:
            return (common_min, common_max)
        else:
            return None

    def _skewness(self, data:np.ndarray) -> float:
        nom = np.sum((data - np.mean(data))**3) / len(data)
        denom = np.var(data)

        return nom / denom**(1.5) 
    
    @property
    def n_sequences(self):
        return self._n_sequences
        
    @n_sequences.setter
    def n_sequences(self, value):
        self._n_sequences = value