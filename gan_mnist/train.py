from GAN import GAN

#make model
gan = GAN()

#train
gan.train(iterations=200000, batch_size=32, save_interval=1000, model_interval=50000, check_noise=gan.make_noise(25), raw=5, col=5)