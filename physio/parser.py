import numpy as np
from scipy.io import wavfile
import os
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from collections import namedtuple
from sklearn.cross_validation import check_random_state

MlData = namedtuple('ml_data', 'X y')


class PCG:
    def __init__(self, basepath, random_state=42):
        self.basepath = basepath
        self.class_name_to_id = {"normal": 0, "abnormal": 1}
        self.train = None
        self.test = None

        self.n_samples = 0

        self.X = None
        self.y = None

        self.random_state = random_state

        self._load_wav_file()
        self._split_train_test()

    def _load_wav_file(self):

        # First pass to calculate number of samples
        for root, dirs, files in os.walk(self.basepath):
            # Ignore validation for now!
            if "validation" in root:
                continue
            for file in files:
                if file.endswith('.wav'):
                    self.n_samples += 1

        # Looking at the number of wav files (3541)
        # And truncating the length of the file to the
        # min file size (10611) (Note: this may be bad
        # And may cause loss of information!)
        X = np.zeros([self.n_samples, 10611])
        file_names = []
        class_labels = []
        i = 0
        for root, dirs, files in os.walk(self.basepath):

            # Ignore validation for now!
            if "validation" in root:
                continue
            for file in files:
                if file.endswith('.wav'):
                    wavfname = os.path.join(root, file)
                    _, wf = wavfile.read(wavfname)
                    X[i, :] = wf[:10611]

                    file_names.append(wavfname)

                    base_file_name = file.rstrip(".wav")
                    label_file_name = os.path.join(root, base_file_name + ".hea")

                    with open(label_file_name, 'r') as fin:
                        header = fin.readlines()

                    comments = [line for line in header if line.startswith("#")]
                    if not len(comments) == 1:
                        raise Exception("Invalid label file %s" % label_file_name)

                    class_label = str(comments[0]).lstrip("#").rstrip("\r").strip().lower()

                    if not class_label in self.class_name_to_id.keys():
                        raise Exception("Invalid class label %s" % class_label)

                    class_labels.append(self.class_name_to_id[class_label])

                    i += 1

        self.X = normalize(X, axis=1)

        class_labels = np.array(class_labels)
        self.y = np.eye(len(self.class_name_to_id.keys()))[class_labels]

    def get_mini_batch(self, batch_size):
        random_state = check_random_state(None)  # self.random_state)
        n_training_samples = self.train.X.shape[0]
        minibatch_indices = random_state.randint(0, n_training_samples - 1, batch_size)

        return self.train.X[minibatch_indices, :], self.train.y[minibatch_indices, :]

    def _split_train_test(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25,
                                                            random_state=self.random_state)

        self.train = MlData(X=X_train, y=y_train)
        self.test = MlData(X=X_test, y=y_test)


if __name__ == '__main__':
    print 'hi'
    # load_wav_file('/datasets/physiobank/2016_challenge')
