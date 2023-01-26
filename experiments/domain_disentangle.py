import torch
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
         # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel(opt)
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
        self.entropyLoss = lambda outputs : -torch.mean(torch.sum(self.logSoftmax(outputs), dim=1))
        self.mseloss = torch.nn.MSELoss()
        self.kldivloss = torch.nn.KLDivLoss(reduction="batchmean")

        self.weights = [0.6, 0.3, 0.1]

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

        self.optimizer1.zero_grad()

        if not self.opt['dom_gen']:
            if domain == 0:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                features, rec_features, source_class_outputs, source_dom_outputs, source_adv_objC_outputs, source_adv_domC_outputs = self.model(images, True)
                source_class_loss = self.weights[0]*self.crossEntropyLoss(source_class_outputs, labels)
                source_dom_loss = self.weights[1]*self.crossEntropyLoss(source_dom_outputs, torch.zeros(source_dom_outputs.size()[0], dtype = torch.long).to(self.device))
                reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
                source_adv_domC_loss = self.weights[0]*self.opt["alpha"]*self.entropyLoss(source_adv_domC_outputs)
                source_adv_objC_loss = self.weights[1]*self.opt["alpha"]*self.entropyLoss(source_adv_objC_outputs)
                total_loss = (source_class_loss + source_adv_domC_loss) + (source_dom_loss + source_adv_objC_loss) + reconstruction_loss
            else:
                images, _ = data
                images = images.to(self.device)
                features, rec_features, _ , target_dom_outputs, target_adv_objC_outputs, target_adv_domC_outputs = self.model(images, True)
                target_dom_loss = self.weights[0]*self.crossEntropyLoss(target_dom_outputs, torch.ones(target_dom_outputs.size()[0], dtype = torch.long).to(self.device))
                reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
                target_adv_domC_loss =  self.weights[0]*self.opt["alpha"]*self.entropyLoss(target_adv_domC_outputs)
                target_adv_objC_loss = self.weights[1]*self.opt['alpha']*self.entropyLoss(target_adv_objC_outputs)
                total_loss = (target_dom_loss + target_adv_domC_loss) + target_adv_objC_loss + reconstruction_loss
        else:
            examples, labels = data
            images, dom_labels = examples
            images = images.to(self.device)
            labels = labels.to(self.device)
            dom_lables = dom_labels.to(self.device)
            features, rec_features, source_class_outputs, source_dom_outputs, source_adv_objC_outputs, source_adv_domC_outputs = self.model(images, True, True)
            source_class_loss = self.weights[0]*self.crossEntropyLoss(source_class_outputs, labels)
            source_dom_loss = self.weights[1]*self.crossEntropyLoss(source_dom_outputs, dom_lables)
            reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
            source_adv_domC_loss = self.weights[0]*self.opt["alpha"]*self.entropyLoss(source_adv_domC_outputs)
            source_adv_objC_loss = self.weights[1]*self.opt["alpha"]*self.entropyLoss(source_adv_objC_outputs)
            total_loss = (source_class_loss + source_adv_domC_loss) + (source_dom_loss + source_adv_objC_loss) + reconstruction_loss
        
        
        total_loss.backward()
        self.optimizer1.step()
        return total_loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for z, y in loader:
                if not self.opt['dom_gen']:
                    x = z
                else:
                    x, _ = z
                    
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x, False)
                loss += self.crossEntropyLoss(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss