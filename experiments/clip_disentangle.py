import torch
from models.base_model import DomainDisentangleModel
import clip

class CLIPDisentangleExperiment: # See point 4. of the project
    
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

        # Setup clip model
        self.clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Setup optimization procedure
        self.optimizer1 = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)
        self.entropyLoss = lambda outputs : -torch.mean(torch.sum(self.logSoftmax(outputs), dim=1))
        self.mseloss = torch.nn.MSELoss()

        self.mseloss_sum = torch.nn.MSELoss(reduction='sum')
        self.l2loss = lambda outputs : torch.sqrt(self.mseloss_sum(outputs))

        self.weights = [0.6, 0.3, 0.1]

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer1'] = self.optimizer1.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer1.load_state_dict(checkpoint['optimizer1'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, domain): 
        #domain==0 -> source
        #domain==1 -> target
        
        images = []
        labels = []
        descriptions = []

        self.optimizer1.zero_grad()

        print("-----------------------------")

        if domain == 0:
            images, labels, descriptions = data
            images = images.to(self.device)
            labels = labels.to(self.device)
            descriptions = descriptions.to(self.device)
            features, rec_features, source_class_outputs, source_dom_outputs, source_adv_objC_outputs, source_adv_domC_outputs = self.model(images, True)
            source_class_loss = self.weights[0]*self.crossEntropyLoss(source_class_outputs, labels)
            #print(f"source_class_loss: {source_class_loss.item()}")
            source_dom_loss = self.weights[1]*self.crossEntropyLoss(source_dom_outputs, torch.zeros(source_dom_outputs.size()[0], dtype = torch.long).to(self.device))
            #print("source_dom_loss: ",source_dom_loss.item())
            reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
            #print("reconstruction_loss: ",reconstruction_loss.item())
            source_adv_domC_loss = self.weights[0]*self.opt["alpha"]*self.entropyLoss(source_adv_domC_outputs)
            source_adv_objC_loss = self.weights[1]*self.opt["alpha"]*self.entropyLoss(source_adv_objC_outputs)
            #print("source_adv_domC_loss: ",source_adv_domC_loss.item())
            #print("source_adv_objC_loss: ",source_adv_objC_loss.item())

            tokenized_text = self.clip_model.tokenize(descriptions).to(self.device)
            text_features = self.clip_model.encode_text(tokenized_text)
            
            clip_loss = self.l2loss(text_features, source_dom_outputs)
            print("clip_loss: ", clip_loss.item())

            total_loss = (source_class_loss + source_adv_domC_loss) + (source_dom_loss + source_adv_objC_loss) + reconstruction_loss + clip_loss
            #print("total_loss: ", total_loss.item())
        else:
            images, _, descriptions = data
            images = images.to(self.device)
            descriptions = descriptions.to(self.device)
            features, rec_features, _ , target_dom_outputs, target_adv_objC_outputs, target_adv_domC_outputs = self.model(images, True)
            target_dom_loss = self.weights[0]*self.crossEntropyLoss(target_dom_outputs, torch.ones(target_dom_outputs.size()[0], dtype = torch.long).to(self.device))
            reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
            #print("reconstruction_loss: ",reconstruction_loss.item())
            target_adv_domC_loss =  self.weights[0]*self.opt["alpha"]*self.entropyLoss(target_adv_domC_outputs)
            target_adv_objC_loss = self.weights[1]*self.opt['alpha']*self.entropyLoss(target_adv_objC_outputs)
            #print("target_dom_loss: ",target_dom_loss.item())
            #print("target_adv_domC_loss: ",target_adv_domC_loss.item())
            #print("target_adv_objC_loss: ",target_adv_objC_loss.item())

            tokenized_text = self.clip_model.tokenize(descriptions).to(self.device)
            text_features = self.clip_model.encode_text(tokenized_text)

            clip_loss = self.l2loss(text_features, target_dom_outputs)
            print("clip_loss: ", clip_loss.item())

            total_loss = (target_dom_loss + target_adv_domC_loss) + target_adv_objC_loss + reconstruction_loss + clip_loss
            #print("total_loss: ", total_loss.item())
        
        
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

                logits = self.model(x, False)
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