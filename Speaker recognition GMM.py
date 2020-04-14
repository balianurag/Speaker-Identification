import librosa
import numpy as np
import os
import sklearn.mixture
import nlpaug.augmenter.audio as naa

np.random.seed(42)

def load(audio_path):
    y, sr = librosa.load(audio_path)
    aug_noise = naa.NoiseAug(noise_factor=sr/1000)
    augmented_noise=aug_noise.substitute(y)
    y_trim = librosa.effects.remix(augmented_noise, intervals=librosa.effects.split(y))
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr)
    return mfcc.T

def fit(frames, test_ratio=0.05, n_components=32):
    index = np.arange(len(frames))
    np.random.shuffle(index)
    train_idx = index[int(len(index) * test_ratio):]
    test_idx = index[:int(len(index) * test_ratio)]
    gmm = sklearn.mixture.GaussianMixture(n_components=n_components)
    gmm.fit(frames[train_idx])
    return gmm, frames[test_idx]

def predict(gmms, test_frame):
    scores = []
    for gmm_name, gmm in gmms.items():
        scores.append((gmm_name, gmm.score(test_frame)))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def evaluate(gmms, test_frames):
    correct = 0
    for name in test_frames:
        best_name, best_score = predict(gmms, test_frames[name])[0]
        print('Ground Truth: %s, Predicted: %s, Score: %d' % (name, best_name, abs(best_score)))    

if __name__ == '__main__':
    gmms, test_frames = {}, {}
    a=os.listdir("The location of dataset of directory")
    print(a)
    file="The same directory address without last '\\' "
    out="Destination Folder if want to save the output "
    test_file="The test file location"

    for i in a[:]:
        filename=os.path.join(file,i)
        name = os.path.splitext(os.path.basename(filename))[0]
        print('Processing %s ...' % name)
        gmms[name], test_frames[name] = fit(load(filename))
        evaluate(gmms, test_frames)
    
    result = predict(gmms, load(test_file))
    
    print("Testing Audio Name :- ",os.path.basename(test_file)," , Identified Audio Name:- ",result[0][0])
