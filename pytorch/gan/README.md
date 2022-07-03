# GANS

## 1. GAN (Multilayer Perceptron)

## 2. DCGAN (Deep Convolutional)

| D(x)                                                                 | D(G(z)) | 1-D(G(z)) | (D(G(z)) + 1-D(G(z)))/2                | Result                                                                        |
|----------------------------------------------------------------------|------|-----------|----------------------------------------|-------------------------------------------------------------------------------|
| <img src="assets/dcgan/d_loss_epoch.svg" alt="D(x)" width="300px" /> |  <img src="assets/dcgan/d_fake_loss_epoch.svg" alt="D(G(z))" width="300px" />    |    <img src="assets/dcgan/g_loss_epoch.svg" alt="1-D(G(z))" width="300px" />       | <img src="assets/dcgan/d_loss_epoch.svg" alt="(D(G(z)) + 1-D(G(z)))/2" width="300px" /> | <img src="assets/dcgan/dcgan_mnist_20epoch.gif" alt="result" width="300px" /> |
| <img src="assets/dcgan/g_loss_celaba.svg" width="300px" />           | <img src="assets/dcgan/d_fake_loss_celaba.svg" width="300px" /> | <img src="assets/dcgan/d_real_loss_celaba.svg" width="300px" /> | <img src="assets/dcgan/d_loss_celaba.svg" width="300px" /> | <img src="assets/dcgan/dcgan_celeba.gif" width="300px" />                     |

## 3. 