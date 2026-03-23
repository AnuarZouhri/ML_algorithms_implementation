import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import lasso_path
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


def split_data(X, Y, m_test, random_seed):
    return train_test_split(X, Y, test_size=m_test, random_state=random_seed)

def compute_scalar(X_train):
    return preprocessing.StandardScaler().fit(X_train)

def pre_process(scaler, X_train, X_test):
    return [scaler.transform(X_train), scaler.transform(X_test)]

def initialize_sets(name_set, offset = 3):
    df = pd.read_csv(name_set, sep = ',')
    # -- remove the data samples with missing values (NaN)
    df = df.dropna()

    # -- print the column names and the first 5 rows of the dataframe
    print(df.columns)
    print('\n')
    print(df.head())
    data = df.values  # removes headers
    m = df.shape[0]
    print("Number of samples:", m)

    Y = data[:m, 2]
    # -- explain why we are excluding ids (and date)
    X = data[:m, 3:]

    # -- print shapes
    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)
    # description of input data X
    pd.DataFrame(X, columns=df.columns[3:], dtype=float).describe()
    return X, Y, m

def check_sets(X_train, Y_train, X_train_scaled, X_test_scaled):
    print(f'X_train size: {X_train.shape}, Y_train size: {Y_train.shape}')
    print(f'X_test size: {X_test.shape}, Y_test size: {Y_test.shape}')
    # -- let's check if the scaler did his job
    print(f'X_train mean: {np.mean(X_train_scaled)}, X_train std: {np.std(X_train_scaled)}')
    print(f'X_test mean: {np.mean(X_test_scaled)}, X_test std: {np.std(X_test_scaled)}')


X, Y, m = initialize_sets('kc_house_data_reduced.csv')

m_train = int(2/3*m)
m_test = m - m_train
random_seed = 234689

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = m_test, random_state = random_seed)
scaler = compute_scalar(X_train)
X_train_scaled , X_test_scaled = pre_process(scaler, X_train, X_test)

LR = linear_model.LinearRegression()

LR.fit(X_train_scaled, Y_train)

Y_train_predicted = LR.predict(X_train_scaled)
Y_test_predicted = LR.predict(X_test_scaled)

w_LR = np.hstack((LR.intercept_, LR.coef_))
print('Linear regression parameters\n:', w_LR)

loss_train = np.mean((Y_train -Y_train_predicted)**2)
loss_test = np.mean((Y_test -Y_test_predicted)**2)

print('---')
# -- print average loss in training data and in test data. Explain the high values of parameters (due to very high house prices)
print(f'Average loss in training data: {loss_train} || {loss_train:.3e}')
print(f'Average loss in test data: {loss_test} || {loss_test:.3e}')
print('---')

# -- print 1 - coefficient of determination in training data and in test data
print(f'1 - R² on training data:   {(1 - LR.score(X_train_scaled, Y_train))}')
print(f'1 - R² on test data: {1 - LR.score(X_test_scaled, Y_test)}')

# -- define the x locations for the groups
ind = np.arange(1, len(LR.coef_) + 1)

# -- define the width of the bars
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(ind, LR.coef_, width, color='r', label='LR')
ax.legend()
plt.xlabel('Coefficient Idx')
plt.ylabel('Coefficient Value')
plt.title('LR Coefficient')
plt.show()


#REGULARIZATION
lasso_lams = np.logspace(0,4, num=100)
lasso_lams, lasso_coefs, _ = lasso_path(X_train_scaled, Y_train, alphas = lasso_lams)

num_folds = 5
kf = KFold(n_splits = num_folds)

loss_train_lasso_kfolds = np.zeros(len(lasso_lams),)
loss_val_lasso_kfolds = np.zeros(len(lasso_lams),)
err_train_lasso_kfolds = np.zeros(len(lasso_lams),)
err_val_lasso_kfolds = np.zeros(len(lasso_lams),)

for i, lam in enumerate(lasso_lams):
    lasso_kfold = Lasso(alpha=lam, tol=0.5)
    for j, (train_index, validation_index) in enumerate(kf.split(X_train)):

        X_train_kfold, X_val_kfold = X_train[train_index], X_train[validation_index]
        Y_train_kfold, Y_val_kfold = Y_train[train_index], Y_train[validation_index]

        # -- data scaling
        scaler_kfold = preprocessing.StandardScaler().fit(X_train_kfold)
        X_train_kfold = scaler_kfold.transform(X_train_kfold)
        X_val_kfold = scaler_kfold.transform(X_val_kfold)

        lasso_kfold.fit(X_train_kfold, Y_train_kfold)

        Y_pred_train = lasso_kfold.predict(X_train_kfold)
        Y_pred_val = lasso_kfold.predict(X_val_kfold)

        loss_train_lasso_kfolds[i] += np.mean((Y_train_kfold - Y_pred_train)**2)
        loss_val_lasso_kfolds[i] += np.mean((Y_val_kfold - Y_pred_val)**2)

        err_train_lasso_kfolds[i] += np.mean((1 - lasso_kfold.score(X_train_kfold, Y_train_kfold))**2)
        err_val_lasso_kfolds[i] += np.mean((1 - lasso_kfold.score(X_val_kfold, Y_val_kfold)) ** 2)

# -- compute the mean => estimate of train and validation losses and errors for each lam
loss_train_lasso_kfolds /= num_folds
loss_val_lasso_kfolds /= num_folds
err_train_lasso_kfolds /= num_folds
err_val_lasso_kfolds /= num_folds


# -- choose the regularization parameter that minimizes the validation loss
lasso_lam_opt = lasso_lams[np.argmin(loss_val_lasso_kfolds)]
print('Best value of the regularization parameter:', lasso_lam_opt)
print('Min loss: ', np.min(loss_val_lasso_kfolds))
# -- logspace(1, 4) = 0.45749
# -- logspace(1, 6) = 0.4525
print('Min (1 - R²) ', np.min(err_val_lasso_kfolds))



