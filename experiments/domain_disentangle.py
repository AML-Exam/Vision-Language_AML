import torch
from models.base_model import DomainDisentangleModel
from itertools import chain

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

        #self.parameters2 = list(self.model.category_encoder.parameters()) + list(self.model.domain_encoder.parameters()) + list(self.model.feature_extractor.parameters()) + list(self.model.reconstructor.parameters())
        
        # Setup optimization procedure
        # forse possiamo usare il gradient descend
        self.optimizer1 = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        #self.optimizer2 = torch.optim.Adam(self.parameters2, lr=opt['lr'])
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)
        self.entropyLoss = lambda outputs : torch.mean(torch.sum(self.logSoftmax(outputs), dim=1))
        self.mseloss = torch.nn.MSELoss()
        self.kldivloss = torch.nn.KLDivLoss(reduction="batchmean")

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer1'] = self.optimizer1.state_dict()
        #checkpoint['optimizer2'] = self.optimizer2.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer1.load_state_dict(checkpoint['optimizer1'])
        #self.optimizer2.load_state_dict(checkpoint['optimizer2'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, domain): 
        #domain==0 -> source
        #domain==1 -> target
        
        images = []
        labels = []
        weigths = [0.6, 0.3, 0.1]

        self.optimizer1.zero_grad()

        print("-----------------------------")

        if domain == 0:
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)
            features, rec_features, source_class_outputs, source_dom_outputs, source_adv_objC_outputs, source_adv_domC_outputs = self.model(images, True, self.opt["alpha"])
            source_class_loss = weigths[0]*self.crossEntropyLoss(source_class_outputs, labels)
            print(f"source_class_loss: {source_class_loss.item()}")
            source_dom_loss = weigths[1]*self.crossEntropyLoss(source_dom_outputs, torch.zeros(self.opt['batch_size'], dtype = torch.long).to(self.device))
            print("source_dom_loss: ",source_dom_loss.item())
            reconstruction_loss = weigths[2]*self.mseloss(rec_features, features)
            print("reconstruction_loss: ",reconstruction_loss.item())
            source_adv_domC_loss = weigths[0]*self.opt["alpha"]*self.entropyLoss(source_adv_domC_outputs)
            source_adv_objC_loss = weigths[1]*self.opt["alpha"]*self.entropyLoss(source_adv_objC_outputs)
            print("source_adv_domC_loss: ",source_adv_domC_loss.item())
            print("source_adv_objC_loss: ",source_adv_objC_loss.item())
            total_loss = (source_class_loss + source_adv_domC_loss) + (source_dom_loss + source_adv_objC_loss) + reconstruction_loss
            print("total_loss: ", total_loss.item())
        else:
            images, _ = data
            images = images.to(self.device)
            features, rec_features, _ , target_dom_outputs, target_adv_objC_outputs, target_adv_domC_outputs = self.model(images, True, self.opt["alpha"])
            target_dom_loss = weigths[0]*self.crossEntropyLoss(target_dom_outputs, torch.ones(target_dom_outputs.size()[0], dtype = torch.long).to(self.device))
            reconstruction_loss = weigths[2]*self.mseloss(rec_features, features)
            print("reconstruction_loss: ",reconstruction_loss.item())
            target_adv_domC_loss =  weigths[0]*self.opt["alpha"]*self.entropyLoss(target_adv_domC_outputs)
            target_adv_objC_loss = weigths[1]*self.opt['alpha']*self.entropyLoss(target_adv_objC_outputs)
            print("target_dom_loss: ",target_dom_loss.item())
            print("target_adv_domC_loss: ",target_adv_domC_loss.item())
            print("target_adv_objC_loss: ",target_adv_objC_loss.item())
            total_loss = (target_dom_loss + target_adv_domC_loss) + target_adv_objC_loss + reconstruction_loss
            print("total_loss: ", total_loss.item())
        
        
        total_loss.backward()
        self.optimizer1.step()
        return total_loss.item()
        #raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x, False, self.opt["alpha"])
                loss += self.crossEntropyLoss(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss

        #raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')





        #if obj_label != None:
        #    obj_label = obj_label.to(self.device) #to be tested
        #dom_label = dom_label.to(self.device)

        #features, obj_class, dom_class, recon_feat, adv_dom_to_obj_class, adv_obj_to_dom_class = self.model(image, obj_label)
        #
        #if obj_label != None:
        #    celoss_obj = self.criterion(obj_class, obj_label)
        #    eloss_dom_to_obj = - self.criterion(adv_dom_to_obj_class, obj_label)
        #else:
        #    celoss_obj = 0
        #    eloss_dom_to_obj = 0
        #celoss_dom = self.criterion(dom_class, dom_label)
        #eloss_obj_to_dom = - self.criterion(adv_obj_to_dom_class, dom_label)
        #
        ##recontructor loss
        #rec_loss = self.mseloss(recon_feat, features) + self.kldiv(recon_feat, features)
#
        #total_loss = celoss_obj + celoss_dom + eloss_dom_to_obj + eloss_obj_to_dom + rec_loss