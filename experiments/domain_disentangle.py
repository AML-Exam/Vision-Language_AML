import torch
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
         # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        # forse possiamo usare il gradient descend
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mseloss = torch.nn.MSELoss()
        self.kldiv = torch.nn.KLDivLoss(reduction="batchmean")

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, target, source):
        target_images, _ = target
        source_images, source_obj_labels = source
        # obj_label messa a None per il target.
        images = images.to(self.device)
        obj_labels = obj_labels.to(self.device)
        dom_labels = dom_labels.to(self.device)
        
        


        if obj_label != None:
            obj_label = obj_label.to(self.device) #to be tested
        dom_label = dom_label.to(self.device)

        features, obj_class, dom_class, recon_feat, adv_dom_to_obj_class, adv_obj_to_dom_class = self.model(image, obj_label)
        
        if obj_label != None:
            celoss_obj = self.criterion(obj_class, obj_label)
            eloss_dom_to_obj = - self.criterion(adv_dom_to_obj_class, obj_label)
        else:
            celoss_obj = 0
            eloss_dom_to_obj = 0
        celoss_dom = self.criterion(dom_class, dom_label)
        eloss_obj_to_dom = - self.criterion(adv_obj_to_dom_class, dom_label)
        
        #recontructor loss
        rec_loss = self.mseloss(recon_feat, features) + self.kldiv(recon_feat, features)

        total_loss = celoss_obj + celoss_dom + eloss_dom_to_obj + eloss_obj_to_dom + rec_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')

    def validate(self, loader):
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')
