import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB

print("Loading data...")
xpath1 = os.path.join(os.getcwd(), 'data', 'x_train.npy')
xpath2 = os.path.join(os.getcwd(), 'data', 'x_test.npy')
ypath1 = os.path.join(os.getcwd(), 'data', 'y_train.npy')
ypath2 = os.path.join(os.getcwd(), 'data', 'y_test.npy')


X_trn_raw = np.load(xpath1)
X_tst_raw = np.load(xpath2)
y_trn_raw = np.load(ypath1).ravel()
y_tst_raw = np.load(ypath2).ravel()

X_trn_raw = [image.flatten() for image in X_trn_raw]
X_tst_raw = [image.flatten() for image in X_tst_raw]

X_trn = [X_trn_raw[i] for i in range(len(X_trn_raw)) if y_trn_raw[i] != 0]
y_trn = [val for val in y_trn_raw if val != 0]
X_tst = [X_tst_raw[i] for i in range(len(X_tst_raw)) if y_tst_raw[i] != 0]
y_tst = [val for val in y_tst_raw if val != 0]
print("Loading complete!")

print("Training model...")
model = MultinomialNB().partial_fit(X_trn_raw, y_trn_raw, np.unique(y_trn_raw))
print("Training complete! \n")

print("Random accuracy:", 100/len(np.unique(y_tst)), '%')
print("Training accuracy:", 100 * model.score(X_trn, y_trn), '%')
print("Testing accuracy:", 100 * model.score(X_tst, y_tst), '%')
