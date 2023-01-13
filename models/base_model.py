import torch
import torch.nn as nn
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x_r = x.squeeze()
        if len(x_r.size()) < 2:
          x_r = x_r.unsqueeze(0)
        return x_r

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear(512, 2)
        self.category_classifier = nn.Linear(512, 7)

        ##
        self.cv = nn.Conv1d(2,1,2)

        self.reconstructor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
        )

    def forward(self, x, domain_label, alpha = None):
        # x = feature, y classification result
        # c = category, d = domain
        x = self.feature_extractor(x)
        c_x = self.category_encoder(x)
        d_x = self.domain_encoder(x)

        if alpha == None:
            ## convolute the concatenated c_x and d_x
            fg = self.cv(torch.cat((c_x,d_x),0))
            r_x = self.reconstructor(fg)

        # domain_label 0 => source, domain_label 1 => target
        if domain_label == 0:
            if alpha == None:
                c_y = self.category_classifier(c_x)
                d_y = self.domain_classifier(d_x)
                return c_y, d_y, x, r_x
            else:
                a_c_y = self.category_classifier(d_x)
                a_d_y = self.domain_classifier(c_x)
                return a_d_y, a_c_y
        else:
            if alpha == None:
                d_y = self.domain_classifier(d_x)
                return d_y, x, r_x
            else:
                a_d_y = self.domain_classifier(c_x)
                return a_d_y
            

        
        #if target_label != None: #?????
        #    c_y = self.category_classifier(c_x)
        #else:
        #    c_y = None
        #d_y = self.domain_classifier(d_x)
        ##adversarial
        #a_c_y = self.category_classifier(d_x) #valori del domain encoder nel category classifier #?????
        #a_d_y = self.domain_classifier(c_x) #valori del category encoder nel domain classifier
        ## convolute the concatenated c_x and d_x
        #fg = self.cv(torch.cat((c_x,d_x),0))
        #r_x = self.reconstructor(fg)
        #return x, c_y, d_y, r_x, a_c_y, a_d_y
        ## ritorniamo le feature estratte e quelle ricostruite per calcolare la reconstruction loss ( x, r_x )
        ## ritorniamo l'output dei classificatori per le altre loss function ( c_y, d_y )
        ## ritorniamo l'output per l'adversarial search ( a_c_y, a_d_y )
