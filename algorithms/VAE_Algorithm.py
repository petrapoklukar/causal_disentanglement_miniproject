#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:07:53 2019

@author: petrapoklukar

Functions for training a VAE with the additional action loss.

"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from scipy.stats import norm
import sys
sys.path.insert(0,'..')
import importlib
import algorithms.EarlyStopping as ES
import torch.utils.data as data

# ---
# ====================== Training functions ====================== #
# ---
class VAE_Algorithm():
    def __init__(self, opt):
        # Save the whole config
        self.opt = opt

        # Training parameters
        self.batch_size = opt['batch_size']
        self.epochs = opt['epochs']
        self.current_epoch = None
        self.loss_fn = opt['loss_fn']
        self.snapshot = opt['snapshot']
        self.console_print = opt['console_print']
        self.lr_schedule = opt['lr_schedule']
        self.init_lr_schedule = opt['lr_schedule']
        self.model = None
        self.vae_optimiser = None
        self.input_dim= opt['input_dim']
        self.compute_loss = self.compute_gaussian_loss if \
            opt['decoder_param'] == 'gaussian' else self.compute_bernoulli_loss
        print(' *- Loss function set to ', self.compute_loss)

        # Beta scheduling
        self.kl_anneal = opt['kl_anneal']
        self.beta_warmup = opt['beta_warmup']
        self.beta = opt['beta_min']
        self.beta_range = opt['beta_max'] - opt['beta_min'] + 1
        self.beta_steps = opt['beta_steps'] - 1
        self.beta_idx = 0
        self.min_epochs = opt['min_epochs'] if 'min_epochs' in opt.keys() else 99
        self.max_epochs = opt['max_epochs'] if 'max_epochs' in opt.keys() else 199

        # Other parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt['device'] = self.device
        print(' *- Chosen device: ', self.device)
        
        torch.manual_seed(opt['random_seed'])
        np.random.seed(opt['random_seed'])
        print(' *- Chosen random seed: ', self.device)
        
        if self.device == 'cuda': torch.cuda.manual_seed(opt['random_seed'])
        self.save_path = self.opt['exp_dir'] + '/' + self.opt['filename']
        self.model_path = self.save_path + '_model.pt'


    def count_parameters(self):
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def plot_snapshot_loss(self):
        """
        Plots epochs vs model losses on only a certain range of epochs.
        """
        plt_data = np.stack(self.epoch_losses)
        plt_labels = ['loss', 'recon loss', 'kl loss']
        for i in range(len(plt_labels)):
            plt.subplot(len(plt_labels),1,i+1)
            plt.plot(np.arange(self.snapshot)+(self.current_epoch//self.snapshot)*self.snapshot,
                     plt_data[self.current_epoch-self.snapshot+1:self.current_epoch+1, i],
                     label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_SnapshotLosses_{0}'.format(self.current_epoch))
        plt.clf()
        plt.close()


    def plot_model_loss(self):
        """Plots epochs vs model loss."""
        plt_data = np.stack(self.epoch_losses)
        plt_labels = ['loss', 'recon loss', 'kl loss']
        for i in range(len(plt_labels)):
            plt.subplot(len(plt_labels),1,i+1)
            plt.plot(np.arange(self.current_epoch+1),
                     plt_data[:, i],
                     label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_Losses')
        plt.clf()
        plt.close()

        # pureKL
        fig, ax = plt.subplots()
        ax.plot(plt_data[:, 2], 'go-', linewidth=3, label='pKL loss')
        ax.plot(plt_data[:, 1], 'bo--', linewidth=2, label='Recon loss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='pKL vs Recon loss')
        plt.savefig(self.save_path + '_KLvsRecLoss')
        plt.close()

    def sq_else_perm(self, img):
        """"""
        return img.squeeze() if self.grayscale else img.permute(1,2,0)        
            
    def plot_grid(self, images, n=5, name="dec"):
        """
        Plots an nxn grid of images of size digit_size. Used to monitor the 
        reconstruction of decoded images.
        """
        digit_size = int(np.sqrt(self.opt['input_dim']/self.opt['input_channels']))
        filename = self.save_path +name + '_checkpointRecon_{0}'.format(self.current_epoch)
        
        if self.opt['input_channels'] == 1:
            figure = np.zeros((digit_size * n, digit_size * n))
        else: 
            figure = np.zeros((digit_size * n, digit_size * n, self.opt['input_channels']))

        # Construct grid of latent variable values
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        # decode for each square in the grid
        counter = 0
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                digit = self.sq_else_perm(images[counter]).detach().cpu().numpy().transpose(1, 2, 0)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
                counter += 1

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='bone')
        plt.savefig(filename)
        plt.clf()
        plt.close()


    def plot_learning_curve(self):
        """Plots train and test learning curves of the VAE training."""
        train_losses_np = np.stack(self.epoch_losses)
        valid_losses_np = np.stack(self.valid_losses)
        assert(len(valid_losses_np) == len(train_losses_np))

        # Non weighted losses
        plt_labels = ['loss', 'recon', 'kl']
        for i in range(len(plt_labels)):
            plt.subplot(len(plt_labels),1,i+1)
            plt.plot(train_losses_np[:, i], 'go-',
                     linewidth=3, label='Train ' + plt_labels[i])
            plt.plot(valid_losses_np[:, i], 'bo--',
                     linewidth=2, label='Valid ' + plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_chpntValidTrainLosses')
        plt.clf()
        plt.close()

        # Validation and train model loss
        fig2, ax2 = plt.subplots()
        plt.plot(train_losses_np[:, 0], 'go-', linewidth=3, label='Train')
        plt.plot(valid_losses_np[:, 0], 'bo--', linewidth=2, label='Valid')
        ax2.plot()
        ax2.set_xlim(0, self.epochs)
        ax2.set(xlabel='# epochs', ylabel='loss', title='Model loss')
        plt.savefig(self.save_path + '_chpntValidTrainModelLoss')
        plt.close()

        # Weighted KL loss
        fig2, ax2 = plt.subplots()
        plt.plot(train_losses_np[:, 3], 'go-', linewidth=3, label='Train')
        plt.plot(valid_losses_np[:, 3], 'bo--', linewidth=2, label='Valid')
        ax2.plot()
        ax2.set_xlim(0, self.epochs)
        ax2.set(xlabel='# epochs', ylabel='loss', title='KL loss')
        plt.savefig(self.save_path + '_chpntValidTrainKLLoss')
        plt.close()


    def compute_gaussian_loss(self, x, dec_mu, dec_logvar, enc_mu, enc_logvar):
        """
        Computes the VAE loss on the training batch given the criterion when the
        likelihood is Gaussian.
        """
        # Reconstruction loss
        HALF_LOG_TWO_PI = 0.91893
        dec_var = torch.exp(dec_logvar)
        batch_rec = torch.sum(
                HALF_LOG_TWO_PI + 0.5 * dec_logvar + 0.5 * ((x - dec_mu) / dec_var) ** 2,
                dim=(1, 2, 3)) # batch_size
        batch_rec = torch.mean(batch_rec)

        # KL loss
        kl_loss = -0.5 * torch.sum(
                (1 + enc_logvar - enc_mu**2 - torch.exp(enc_logvar)),
                dim=1) # batchsize
        batch_kl = torch.mean(kl_loss) # + KL term for minimization
        return batch_rec + self.beta * batch_kl, batch_rec, batch_kl
    
    
    def compute_bernoulli_loss(self, x, dec_mu, dec_logvar, enc_mu, enc_logvar):
        """
        Computes the VAE loss on the training batch given the criterion when the
        likelihood is Bernoulli.
        """
        criterion = torch.nn.BCELoss()
        batch_rec = criterion(dec_mu.view(-1, self.input_dim), 
                              x.view(-1, self.input_dim)) * float(self.input_dim)
        
        # KL loss
        kl_loss = -0.5 * torch.sum(
                (1 + enc_logvar - enc_mu**2 - torch.exp(enc_logvar)),
                dim=1) # batchsize
        batch_kl = torch.mean(kl_loss) # + KL term for minimization
        return batch_rec + self.beta * batch_kl, batch_rec, batch_kl
        
        
    def compute_loss(self, x, dec_mu, dec_logvar, enc_mu, enc_logvar):
        """
        Computes the usual VAE loss on the training batch depending on the 
        parametrisation of the likelihood function.
        """
        return self.compute_gaussian_loss if self.opt['decoder_param'] == 'gaussian' \
            else self.compute_bernoulli_loss


    def compute_test_loss(self, valid_dataset):
        """Computes the VAE loss on the a batch."""
        self.model.eval()
        assert(not self.model.training)

        batch_size = min(len(valid_dataset), self.batch_size)
        valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size, drop_last=True)

        losses = np.zeros(4)
        
        for batch_idx, img in enumerate(valid_dataloader):
            img = img.to(self.device)
            
            # VAE loss on img1
            dec_mean, dec_logvar, enc_mean, enc_logvar = self.model(img)
            the_loss, rec_loss, kl_loss = self.compute_loss(
                    img, dec_mean, dec_logvar, enc_mean, enc_logvar)

            losses += self.format_loss([the_loss, rec_loss, kl_loss])

        n_valid = len(valid_dataloader)
        return losses / n_valid


    def format_loss(self, losses_list):
        """Rounds the loss and returns an np array for logging."""
        reformatted = list(map(lambda x: round(x.item(), 2), losses_list))
        reformatted.append(int(self.current_epoch))
        return np.array(reformatted)


    def init_model(self):
        """Initialises the VAE model."""
        vae = importlib.import_module("architectures.{0}".format(self.opt['model']))
        print(' *- Imported module: ', vae)
        try:
            class_ = getattr(vae, self.opt['model'])
            instance = class_(self.opt).to(self.device)
            return instance
        except:
            raise NotImplementedError(
                    'Model {0} not recognized'.format(self.opt['model']))


    def init_optimiser(self):
        """Initialises the optimiser."""
        print(self.model.parameters())
        if self.opt['optim_type'] == 'Adam':
            print(' *- Initialised Adam optimiser.')
            vae_optim = optim.Adam(self.model.parameters(), lr=self.lr)
            return vae_optim
        else:
            raise NotImplementedError(
                    'Optimiser {0} not recognized'.format(self.opt['optim_type']))


    def update_learning_rate(self, optimiser):
        """Annealing schedule for the learning rate."""
        if self.current_epoch == self.lr_update_epoch:
            for param_group in optimiser.param_groups:
                self.lr = self.new_lr
                param_group['lr'] = self.lr
                print(' *- Learning rate updated - new value:', self.lr)
                try:
                    self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
                except:
                    print(' *- Reached the end of the update schedule.')
                print(' *- Remaning lr schedule:', self.lr_schedule)


    def update_beta(self):
        """Annealing schedule for the KL term."""
        if self.kl_anneal and self.current_epoch >= self.beta_warmup:        	
	        beta_current_step = (self.beta_idx + 1.0) / self.beta_steps
	        epoch_to_update = beta_current_step * self.epochs
	        if self.current_epoch > epoch_to_update and beta_current_step <= 1:
	            self.beta = beta_current_step * self.beta_range
	            self.beta_idx += 1
	            print (' *- Beta updated - new value:', self.beta)


    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):
        """Trains a model with given hyperparameters."""
        # # Debugging & Testing
        # import torch.utils.data as data
        # train_sampler = data.SubsetRandomSampler(
        #     np.random.choice(list(range(len(train_dataset))),
        #                     64, replace=False))
        # # TODO: shuffle, sampler

        dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=num_workers, drop_last=True)#, sampler=train_sampler)
        n_data = len(train_dataset)
        assert(train_dataset.dataset_name == test_dataset.dataset_name)

        print(('\nPrinting model specifications...\n' +
               ' *- Path to the model: {0}\n' +
               ' *- Training dataset: {1}\n' +
               ' *- Number of training samples: {2}\n' +
               ' *- Number of epochs: {3}\n' +
               ' *- Loss criterion: {4}\n' +
               ' *- Batch size: {5}\n'
               ).format(self.model_path, train_dataset.dataset_name, n_data,
                   self.epochs, self.loss_fn, self.batch_size))

        if chpnt_path:
            # Pick up the last epochs specs
            self.load_checkpoint(chpnt_path)

        else:
            # Initialise the model
            self.model = self.init_model()
            self.start_epoch, self.lr = self.lr_schedule.pop(0)
            try:
                self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
            except:
                self.lr_update_epoch, self.new_lr = self.start_epoch - 1, self.lr
            self.vae_optimiser = self.init_optimiser()
            self.valid_losses = []
            self.epoch_losses = []       

            print((' *- Learning rate: {0}\n' +
                   ' *- Next lr update at {1} to the value {2}\n' +
                   ' *- Remaining lr schedule: {3}'
                   ).format(self.lr, self.lr_update_epoch, self.new_lr,
                   self.lr_schedule))

        es = ES.EarlyStopping(patience=300)
        num_parameters = self.count_parameters()
        self.opt['num_parameters'] = num_parameters
        print(' *- Model parameter/training samples: {0}'.format(
                num_parameters/len(train_dataset)))
        print(' *- Model parameters: {0}'.format(num_parameters))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                spacing = 1
                print('{0:>2}{1}\n\t of dimension {2}'.format('', name, spacing),
                      list(param.shape))

        print('\nStarting to train the model...\n' )
        for self.current_epoch in range(self.start_epoch, self.epochs):         
            # Update hyperparameters
            self.model.train()
            self.update_beta()
            self.update_learning_rate(self.vae_optimiser)
            epoch_loss = np.zeros(4)
            for batch_idx, img in enumerate(dataloader):
                img = img.to(self.device)
                
                # VAE loss on img
                dec_mean, dec_logvar, enc_mean, enc_logvar = self.model(img)
                the_loss, rec_loss, kl_loss = self.compute_loss(
                        img, dec_mean, dec_logvar, enc_mean, enc_logvar)

                # Optimise the VAE for the complete loss
                self.vae_optimiser.zero_grad()
                the_loss.backward()
                self.vae_optimiser.step()

                # Monitoring the learning
                epoch_loss += self.format_loss([the_loss, rec_loss, kl_loss])

            # Monitor the training error
            epoch_loss /= len(dataloader)
            self.epoch_losses.append(epoch_loss)
            self.plot_model_loss()
            
            # Monitor the test error
            valid_loss = self.compute_test_loss(test_dataset)
            self.valid_losses.append(valid_loss)
            self.plot_learning_curve()

            # Check that the at least 350 epochs are done
            if (es.step(valid_loss[0]) and self.current_epoch > self.min_epochs) or \
                self.current_epoch > self.max_epochs or torch.isnan(the_loss):
                break

            # Update the checkpoint only if no early stopping was done
            self.save_checkpoint(epoch_loss[0])
            
            if self.current_epoch in (1, 3, 6):
                 self.save_checkpoint(epoch_loss[0], keep=True)

            # Print current loss values every epoch
            if (self.current_epoch + 1) % self.console_print == 0:
                print('Epoch {0}:'.format(self.current_epoch))
                print('   Train loss: {0:.3f} recon loss: {1:.3f} KL loss: {2:.3f}'.format(
                        epoch_loss[0], epoch_loss[1], epoch_loss[2]))
                print('   Valid loss: {0:.3f} recon loss: {1:.3f} KL loss: {2:.3f}'.format(
                        valid_loss[0], valid_loss[1], valid_loss[2]))
                print('   Beta: {0:.6e}'.format(self.beta))
                print('   LR: {0:.6e}\n'.format(self.lr))
            
            # Print validation results when specified
            if (self.current_epoch + 1) % self.snapshot == 0:

                # Plot reconstructions
                self.plot_grid(dec_mean)
                self.plot_grid(img, name="input")
                self.model.eval()

                # Plot training and validation loss
                self.save_checkpoint(epoch_loss[0], keep=True)

                # Write logs
                self.save_logs(train_dataset, test_dataset)
                self.plot_snapshot_loss()

        print('Training completed.')
        self.plot_model_loss()
        self.model.eval()

        # Save the model
        torch.save(self.model.state_dict(), self.model_path)
        self.save_logs(train_dataset, test_dataset)


    def save_logs(self, train_dataset, test_dataset):
        """Saves all the logs to a file."""
        log_filename = self.save_path + '_logs.txt'
        valid_losses = np.stack(self.valid_losses)
        epoch_losses = np.stack(self.epoch_losses)

        with open(log_filename, 'w') as f:
            f.write('Model {0}\n\n'.format(self.opt['filename']))
            f.write( str(self.opt) )
            f.writelines(['\n\n',
                    '*- Model path: {0}\n'.format(self.model_path),
                    '*- Training dataset: {0}\n'.format(train_dataset.dataset_name),
                    '*- Number of training examples: {0}\n'.format(len(train_dataset)),
                    '*- Model parameters/Training examples ratio: {0}\n'.format(
                            self.opt['num_parameters']/len(train_dataset)),
                    '*- Number of testing examples: {0}\n'.format(len(test_dataset)),
                    '*- Learning rate schedule: {0}\n'.format(self.init_lr_schedule),
                    ])
            f.write('*- Train/validation model_loss:\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.2f}/{2:.2f}\n'.format('', t, v, e),
                    epoch_losses[:, 0], valid_losses[:, 0], epoch_losses[:, -1])))

            f.write('*- Train/validation recon_loss:\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.2f}/{2:.2f}\n'.format('', t, v, e),
                    epoch_losses[:, 1], valid_losses[:, 1], epoch_losses[:, -1])))

            f.write('*- Train/validation kl_loss:\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.2f}/{2:.2f}\n'.format('', t, v, e),
                    epoch_losses[:, 2], valid_losses[:, 2], epoch_losses[:, -1])))
        print(' *- Model saved.\n')


    def save_checkpoint(self, epoch_ml, keep=False):
        """Saves a checkpoint during the training."""
        if keep:
            path = self.save_path + '_checkpoint{0}.pth'.format(self.current_epoch)
            checkpoint_type = 'epoch'
        else:
            path = self.save_path + '_lastCheckpoint.pth'
            checkpoint_type = 'last'

        training_dict = {
                'last_epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'vae_optimiser_state_dict': self.vae_optimiser.state_dict(),
                'last_epoch_loss': epoch_ml,
                'valid_losses': self.valid_losses,
                'epoch_losses': self.epoch_losses,
                'beta': self.beta,
                'beta_range': self.beta_range,
                'beta_steps': self.beta_steps,
                'beta_idx': self.beta_idx,
                'snapshot': self.snapshot,
                'console_print': self.console_print,
                'current_lr': self.lr,
                'lr_update_epoch': self.lr_update_epoch,
                'new_lr': self.new_lr,
                'lr_schedule': self.lr_schedule
                }
        torch.save({**training_dict, **self.opt}, path)
        print(' *- Saved {1} checkpoint {0}.'.format(self.current_epoch, checkpoint_type))


    def load_checkpoint(self, path, eval=False):
        """
        Loads a checkpoint and initialises the models to continue training.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self.init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.lr = checkpoint['current_lr']
        self.lr_update_epoch = checkpoint['lr_update_epoch']
        self.new_lr = checkpoint['new_lr']
        self.lr_schedule = checkpoint['lr_schedule']
        self.vae_optimiser= self.init_optimiser()
        self.vae_optimiser.load_state_dict(checkpoint['vae_optimiser_state_dict'])

        self.start_epoch = checkpoint['last_epoch'] + 1
        self.snapshot = checkpoint['snapshot']
        self.valid_losses = checkpoint['valid_losses']
        self.epoch_losses = checkpoint['epoch_losses']

        self.beta = checkpoint['beta']
        self.beta_range = checkpoint['beta_range']
        self.beta_steps = checkpoint['beta_steps']
        self.beta_idx = checkpoint['beta_idx']
        self.current_epoch = checkpoint['last_epoch']

        self.snapshot = checkpoint['snapshot']
        self.console_print = checkpoint['console_print']

        print(('\nCheckpoint loaded.\n' +
               ' *- Last epoch {0} with loss {1}.\n'
               ).format(checkpoint['last_epoch'],
               checkpoint['last_epoch_loss']))
        print(' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                self.lr, self.lr_update_epoch, self.new_lr)
              )
        if eval == False:
            self.model.train()
        else:
            self.model.eval()




# ---
# ====================== Train the models ====================== #
# ---
if __name__ == '__main___':
    import pickle
    class TripletTensorDataset(data.Dataset):
        def __init__(self, dataset_name, split):
            self.split = split.lower()
            self.dataset_name =  dataset_name.lower()
            self.name = self.dataset_name + '_' + self.split

            # Test on causal_data
            if self.dataset_name == 'causal_shapes':
                if split == 'test':
                    with open('../datasets/test_causal_shapes.pkl', 'rb') as f:
                        self.data = pickle.load(f)
                else:
                    with open('../datasets/train_causal_shapes.pkl', 'rb') as f:
                        self.data = pickle.load(f)
            else:
                raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))

        def __getitem__(self, index):
            img = self.data[index]
            return img

        def __len__(self):
            return len(self.data)


    train_dataset = TripletTensorDataset('causal_shapes', 'train')
    test_dataset = TripletTensorDataset('causal_shapes', 'test')

    vae_opt = {
        'model': 'VAE_ResNet', # class name
        'filename': 'vae',
        'exp_dir': 'DUMMY',

        'loss_fn': 'learnable full gaussian',
        'num_workers': 2,
        'device': 'cpu',
        'input_channels': 3,
        'latent_dim': 64,
        'out_activation': 'sigmoid',
        'dropout': 0.1,
        'weight_init': 'normal_init',
        'input_dim': 32*32*3,

        'conv1_out_channels': 32,
        'kernel_size': 3,
        'num_scale_blocks': 3,
        'block_per_scale': 1,
        'depth_per_block': 2,
        'fc_dim': 512,
        'image_size': 32,
        'learn_dec_logvar': True,

        'epochs': 30,
        'batch_size': 25,
        'lr_schedule': [(0, 1e-5), (7, 5e-3)], 
        'snapshot': 5,
        'console_print': 1,
        'beta_min': 0,
        'beta_max': 1,
        'beta_steps': 1,
        'kl_anneal': True,

        'optim_type': 'Adam',
        'random_seed': 1201
    }

    algorithm = VAE_Algorithm(vae_opt)

