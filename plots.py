'''
    create plot util funcs
'''

import matplotlib.pyplot as plt

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()

def plot_pca(x_pca, y, current):
    plt.figure(figsize=(12,12))
    plt.scatter(x_pca[y==0, 0], x_pca[y==0, 1], color='red', alpha=0.5,label='0')
    plt.scatter(x_pca[y==1, 0], x_pca[y==1, 1], color='blue', alpha=0.5,label='1')
    plt.scatter(x_pca[y==2, 0], x_pca[y==2, 1], color='green', alpha=0.5,label='2')
    plt.scatter(x_pca[y==3, 0], x_pca[y==3, 1], color='black', alpha=0.5,label='3')
    plt.scatter(x_pca[y==4, 0], x_pca[y==4, 1], color='khaki', alpha=0.5,label='4')
    plt.scatter(x_pca[y==5, 0], x_pca[y==5, 1], color='yellow', alpha=0.5,label='5')
    plt.scatter(x_pca[y==6, 0], x_pca[y==6, 1], color='turquoise', alpha=0.5,label='6')
    plt.scatter(x_pca[y==7, 0], x_pca[y==7, 1], color='pink', alpha=0.5,label='7')
    plt.scatter(x_pca[y==8, 0], x_pca[y==8, 1], color='moccasin', alpha=0.5,label='8')
    plt.scatter(x_pca[y==9, 0], x_pca[y==9, 1], color='olive', alpha=0.5,label='9')
    plt.scatter(x_pca[y==10, 0], x_pca[y==10, 1], color='coral', alpha=0.5,label='10')
    plt.title("PCA " + current)
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend()
    plt.show()
