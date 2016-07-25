import os
import pickle
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from collections import namedtuple
from sklearn.cross_validation import check_random_state


class PCG:
    """
    PCG is a container for loading phonocardiogram (PCG) data for the [2016 physionet
    challenge](http://physionet.org/challenge/2016). Raw wav files are parsed into
    features, class labels are extracted from header files and data is split into
    training and testing groups.
    """
    def __init__(self, basepath, random_state=42):
        self.basepath = basepath
        self.class_name_to_id = {"normal": 0, "abnormal": 1}
        self.nclasses = len(self.class_name_to_id.keys())

        self.train = None
        self.test = None

        self.n_samples = 0

        self.X = None
        self.y = None

        self.random_state = random_state

    def initialize_wav_data(self):
        """
        Load the original wav files and extract features. Warning, this can take a
        while due to slow FFTs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.__load_wav_file()
        self.__split_train_test()
        # TODO: check if directory exists
        self.save("/tmp")

    def save(self, save_path):
        """
        Persist the PCG class to disk

        Parameters
        ----------
        save_path: str
            Location on disk to store the parsed PCG metadata

        Returns
        -------
        None

        """
        np.save(os.path.join(save_path, "X.npy"), self.X)
        np.save(os.path.join(save_path, "y.npy"), self.y)
        with open( os.path.join(save_path, "meta"), "w") as fout:
            pickle.dump((self.basepath, self.class_name_to_id, self.nclasses,
                         self.n_samples, self.random_state), fout)

    def load(self, load_path):
        """
        Load a previously stored PCG class.

        Parameters
        ----------
        load_path: str
            Location on disk to load parsed PCG data

        Returns
        -------
        None

        """
        self.X = np.load(os.path.join(load_path, "X.npy"))
        self.y = np.load(os.path.join(load_path, "y.npy"))
        with open(os.path.join(load_path, "meta"), "r") as fin:
            (self.basepath, self.class_name_to_id, self.nclasses,
             self.n_samples, self.random_state) = pickle.load(fin)
        self.__split_train_test()

    def __load_wav_file(self, doFFT=True):
        """
        Loads physio 2016 challenge dataset from self.basepath by crawling the path.
        For each discovered wav file:

        * Attempt to parse the header file for class label
        * Attempt to load the wav file
        * Calculate features from the wav file. if doFFT, features are
        the Fourier transform of the original signal. Else, features are
        the raw signal itself truncated to a fixed length

        Parameters
        ----------
        doFFT: bool
            True if features to be calculated are the FFT of the original signal

        Returns
        -------
        None
        """

        # First pass to calculate number of samples
        # ensure each wav file has an associated and parsable
        # Header file
        wav_file_names = []
        class_labels = []
        for root, dirs, files in os.walk(self.basepath):
            # Ignore validation for now!
            if "validation" in root:
                continue
            for file in files:
                if file.endswith('.wav'):
                    try:
                        base_file_name = file.rstrip(".wav")
                        label_file_name = os.path.join(root, base_file_name + ".hea")

                        class_label = self.__parse_class_label(label_file_name)
                        class_labels.append(self.class_name_to_id[class_label])
                        wav_file_names.append(os.path.join(root, file))

                        self.n_samples += 1
                    except InvalidHeaderFileException as e:
                        print e

        if doFFT:
            fft_embedding_size = 400
            highpass_embedding_size = 200
            X = np.zeros([self.n_samples, fft_embedding_size + highpass_embedding_size])
        else:
            # Truncating the length of each wav file to the
            # min file size (10611) (Note: this is bad
            # And causes loss of information!)
            embedding_size = 10611
            X = np.zeros([self.n_samples, embedding_size])

        for idx, wavfname in enumerate(wav_file_names):
            rate, wf = wavfile.read(wavfname)
            wf = normalize(wf.reshape(1, -1))

            if doFFT:
                # We only care about the magnitude of each frequency
                wf_fft = np.abs(fft(wf))
                wf_fft = wf_fft[:, :fft_embedding_size].reshape(-1)

                # Filter out high frequencies via Butter transform
                # The human heart maxes out around 150bpm = 2.5Hz
                # Let's filter out any frequency significantly above this
                nyquist = 0.5 * rate
                cutoff_freq = 4.0  # Hz
                w0, w1 = butter(5, cutoff_freq / nyquist, btype='low', analog=False)
                wf_low_pass = lfilter(w0, w1, wf)

                # FFT the filtered signal
                wf_low_pass_fft = np.abs(fft(wf_low_pass))
                wf_low_pass_fft = wf_low_pass_fft[:, :highpass_embedding_size].reshape(-1)

                features = np.concatenate((wf_fft, wf_low_pass_fft))
            else:
                features = wf[:embedding_size]

            X[idx, :] = features
            idx += 1

        self.X = X

        class_labels = np.array(class_labels)

        # Map from dense to one hot
        self.y = np.eye(self.nclasses)[class_labels]

    def __parse_class_label(self, label_file_name):
        """
        Parses physio bank header files, where the class label
        is located in the last line of the file. An example header
        file could contain:

        f0112 1 2000 60864
        f0112.wav 16+44 1 16 0 0 0 0 PCG
        # Normal


        Parameters
        ----------
        label_file_name: str
            Path to a specific header file

        Returns
        -------
        class_label: str
            One of `normal` or `abnormal`
        """
        with open(label_file_name, 'r') as fin:
            header = fin.readlines()

        comments = [line for line in header if line.startswith("#")]
        if not len(comments) == 1:
            raise InvalidHeaderFileException("Invalid label file %s" % label_file_name)

        class_label = str(comments[0]).lstrip("#").rstrip("\r").strip().lower()

        if not class_label in self.class_name_to_id.keys():
            raise InvalidHeaderFileException("Invalid class label %s" % class_label)

        return class_label

    def __split_train_test(self):
        """
        Splits internal features (self.X) and class labels (self.y) into
        balanced training and test sets using sklearn's helper function.

        Notes:
         * if self.random_state is None, splits will be randomly seeded
           otherwise, self.random_state defines the random seed to deterministicly
           split training and test data
         * For now, class balancing is done by subsampling the overrepresented class.
          Ideally this would be pushed down to the cost function in TensorFlow.

        Returns
        -------
        None
        """
        mlData = namedtuple('ml_data', 'X y')

        num_pos, num_neg = np.sum(self.y, axis=0)

        # Remove samples to rebalance classes
        # TODO: push this down into the cost function
        undersample_rate = num_neg / num_pos
        over_represented_idxs = self.y[:, 1] == 0
        under_represented_idxs = self.y[:, 1] == 1
        random_indexes_to_remove = np.random.rand(self.y.shape[0]) < undersample_rate
        sample_idxs = (over_represented_idxs & random_indexes_to_remove |
                       under_represented_idxs)

        X_balanced = self.X[sample_idxs, :]
        y_balanced = self.y[sample_idxs, :]

        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.25,
                                                            random_state=self.random_state)

        self.train = mlData(X=X_train, y=y_train)
        self.test = mlData(X=X_test, y=y_test)

    def get_mini_batch(self, batch_size):
        """
        Helper function for sampling mini-batches from the training
        set. Note, random_state needs to be set to None or the same
        mini batch will be sampled eternally!

        Parameters
        ----------
        batch_size: int
            Number of elements to return in the mini batch

        Returns
        -------
        X: np.ndarray
            A feature matrix subsampled from self.train

        y: np.ndarray
            A one-hot matrix of class labels subsampled from self.train
        """
        random_state = check_random_state(None)  # self.random_state)
        n_training_samples = self.train.X.shape[0]
        minibatch_indices = random_state.randint(0, n_training_samples - 1, batch_size)

        return self.train.X[minibatch_indices, :], self.train.y[minibatch_indices, :]


class InvalidHeaderFileException(Exception):
    def __init__(self, *args, **kwargs):
        super(args, kwargs)
