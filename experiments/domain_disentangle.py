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
            print(param)

        self.parameters2 = self.model.category_encoder.parameters().append(self.model.domain_encoder.parameters().append(self.model.feature_extractor.parameters().append(self.model.reconstructor.parameters())))

        #debugging
        print([_ for _ in self.model.parameters()])
        print([_ for _ in self.parameters2])
        
        # Setup optimization procedure
        # forse possiamo usare il gradient descend
        #self.optimizer1 = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        #self.optimizer2 = torch.optim.Adam(self.parameters2, lr=opt['lr'])
        #self.crossEntropyLoss = torch.nn.CrossEntropyLoss()
        #self.logSoftmax = torch.nn.LogSoftmax(dim=1)
        #self.entropyLoss = lambda outputs : -torch.mean(torch.sum(self.logSoftmax(outputs), dim=1))
        #self.mseloss = torch.nn.MSELoss()
        #self.kldivloss = torch.nn.KLDivLoss(reduction="batchmean")

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
        source_images, source_labels = source

        target_images = target_images.to(self.device)
        source_images = source_images.to(self.device)
        source_labels = source_labels.to(self.device)

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        # 0 is source domain, 1 is target domain

        ## Direct part
        # Source path
        source_class_outputs, source_dom_outputs, features, rec_features = self.model(source_images, 0)
        source_class_loss = self.crossEntropyLoss(source_class_outputs, source_labels)
        source_dom_loss = self.crossEntropyLoss(source_dom_outputs, torch.zeros(self.opt['batch_size'], dtype = torch.long).to(self.device))
        reconstruction_loss = self.mseloss(rec_features, features) + self.kldivloss(rec_features, features)
        source_partial_loss = source_class_loss + source_dom_loss + reconstruction_loss
        source_partial_loss.backward()
        # Target path
        target_dom_outputs, features, rec_features = self.model(target_images, 1)
        target_dom_loss = self.crossEntropyLoss(target_dom_outputs, torch.ones(self.opt['batch_size'], dtype = torch.long).to(self.device))
        reconstruction_loss = self.mseloss(rec_features, features) + self.kldivloss(rec_features, features)
        target_partial_loss = target_dom_loss + reconstruction_loss
        target_partial_loss.backward()

        self.optimizer1.step()

        ## Adversarial part
        # Source adv path
        source_adv_domC_outputs, source_adv_objC_outputs = self.model(source_images, 0, self.opt['alpha'])
        source_adv_domC_loss = self.entropyLoss(source_adv_domC_outputs)
        source_adv_objC_loss = self.entropyLoss(source_adv_objC_outputs)
        source_adv_partial_loss = self.opt['alpha']*(source_adv_domC_loss + source_adv_objC_loss)
        source_adv_partial_loss.backward()
        # Target adv path
        target_adv_domC_outputs = self.model(target_images, 1, self.opt['alpha'])
        target_adv_domC_loss =  self.opt['alpha']*self.entropyLoss(target_adv_domC_outputs)
        target_adv_domC_loss.backward()

        self.optimizer2.step()

        total_loss = source_partial_loss + target_partial_loss + source_adv_partial_loss + target_adv_domC_loss
        
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

                logits = self.model(x, 0)[0]
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