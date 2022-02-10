import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import math
import matplotlib.pyplot as plt
import util
import Calibration as Calibration


class Model:
    def __init__(self, device, Net, optimizer_Net, optimizer_Calibration, calibration_parameters, num_cali_para, first_train=True, model_path=None):
        self.device = device
        self.Net = Net.to(device)
        self.optimizer_Net = optimizer_Net
        self.optimizer_Calibration = optimizer_Calibration
        self.calibration_parameters = calibration_parameters
        self.num_cali_para = num_cali_para
        self.model_path = model_path

        self.epoch = 0
        self.train_angle_error = []
        self.validation = {
            'best_epoch_cali': [],
            'test_angle_error_1000': [],
            'test_angle_error_2000': []
        }

        if not first_train:
            self.load(model_path)

    def load(self, model_path):
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=self.device)
            self.Net.load_state_dict(model['Net'])
            self.optimizer_Net.load_state_dict(model['optimizer_Net'])
            self.optimizer_Calibration.load_state_dict(model['optimizer_Calibration'])

            self.epoch = model['epoch']
            self.train_angle_error = model['train_angle_error']
            self.validation = model['validation']
        else:
            raise FileNotFoundError('Model file not found!')

    def save(self):
        model = {
            'Net': self.Net.state_dict(),
            'optimizer_Net': self.optimizer_Net.state_dict(),
            'optimizer_Calibration': self.optimizer_Calibration.state_dict(),
            'epoch': self.epoch,
            'train_angle_error': self.train_angle_error,
            'validation': self.validation
        }
        torch.save(model, self.model_path)

    def train(self, epochs, train_loader, test):
        for i in range(self.epoch + 1, self.epoch + 1 + epochs):
            self.Net.train()
            total_angle_error = 0
            for batch_idx, (data, target, index) in enumerate(train_loader):
                data, target, parameters = Variable(data).to(self.device), Variable(target).to(self.device), Calibration.index2parameters(self.calibration_parameters, index).to(self.device)
                AL = self.Net.forward(data, parameters)
                loss = F.mse_loss(AL, target)
                angle_error = util.angle_calculation_avg(AL.cpu().detach().numpy(), target.cpu().detach().numpy())
                total_angle_error += angle_error
                self.optimizer_Net.zero_grad()
                self.optimizer_Calibration.zero_grad()
                loss.backward()
                self.optimizer_Net.step()
                self.optimizer_Calibration.step()
            for params in self.optimizer_Net.param_groups:
                params['lr'] = 0.01 * math.pow(0.1, math.floor((self.epoch + 1) / 35))
            for params in self.optimizer_Calibration.param_groups:
                params['lr'] = 0.1 * math.pow(0.1, math.floor((self.epoch + 1) / 35))
            total_angle_error /= batch_idx + 1
            self.train_angle_error.append(total_angle_error)
            validation = self.validate(test['test_cali_loader'], test['test_test_loader'])
            self.validation['best_epoch_cali'].append(validation['best_epoch_cali'])
            self.validation['test_angle_error_1000'].append(validation['test_angle_error_1000'])
            self.validation['test_angle_error_2000'].append(validation['test_angle_error_2000'])
        self.epoch += i
        self.save()

    def validate(self, test_cali_loader, test_test_loader):
        self.Net.eval()
        calibration_parameter = Calibration.create(1, self.num_cali_para, self.num_cali_para)
        optimizer_Calibration = torch.optim.SGD([calibration_parameter], lr=0.1)
        scale = 100
        epoch = 0
        calibration_parameters = [calibration_parameter.clone()]
        for batch_idx, (data, target, index) in enumerate(test_cali_loader):
            data, target, parameters = Variable(data).to(self.device), Variable(target).to(self.device), Calibration.stack_parameters(calibration_parameter, len(index)).to(self.device)
            data, target, parameters = torch.cat((data, data), 0), torch.cat((target, target), 0), torch.cat((parameters, parameters), 0)
        epoch_scale, test_angle_error_1000, test_angle_error_2000 = None, None, None
        while epoch < 1000:
            AL = self.Net.forward(data, parameters)
            loss = F.mse_loss(AL, target)
            optimizer_Calibration.zero_grad()
            loss.backward()
            optimizer_Calibration.step()
            epoch += 1
            if epoch % scale == 0:
                epoch_scale = math.floor(epoch / scale)
                calibration_parameters.append(calibration_parameter.clone())
                diff = calibration_parameters[-1][0] - calibration_parameters[-2][0]
                L2 = torch.sqrt(torch.sum(diff * diff).data)
                if epoch == 1000:
                    test_angle_error_1000 = self.test(test_test_loader, calibration_parameters[-1].reshape(1, -1))
                if epoch == 2000:
                    test_angle_error_2000 = self.test(test_test_loader, calibration_parameters[-1].reshape(1, -1))
        return {
            'best_epoch_cali': epoch_scale * scale,
            'test_angle_error_1000': test_angle_error_1000,
            'test_angle_error_2000': test_angle_error_2000
        }

    def test(self, loader, calibration_parameters):
        """
        return the mean angle error of the data from one candidate.
        :param loader: container of the data
        :param calibration_parameters: the calibration parameters for this candidate
        :return: mean angle error
        """
        self.Net.eval()
        total_angle_error = 0
        for batch_idx, (data, target, index) in enumerate(loader):
            data, target, parameters = Variable(data).to(self.device), Variable(target).to(self.device), Calibration.stack_parameters(calibration_parameters, len(index)).to(self.device)
            AL = self.Net.forward(data, parameters)
            angle_error = util.angle_calculation_avg(AL.cpu().detach().numpy(), target.cpu().detach().numpy())
            total_angle_error += angle_error

        return total_angle_error / (batch_idx + 1)

