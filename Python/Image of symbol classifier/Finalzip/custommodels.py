import torch.nn as nn

class CNN3train(nn.Module):
    def __init__(self, classes, datatype):
        super(CNN3train, self).__init__()
        self.classes = classes
        self.datatype = datatype
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 100, 3, padding=1),  
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, padding=1),  
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, padding=1), 
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(100, 200, 3, padding=1),  
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3, padding=1),  
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3, padding=1), 
            nn.BatchNorm2d(200),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(200, 300, 3, padding=1), 
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3, padding=1), 
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3, padding=1), 
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(300, 400, 3, padding=1),  
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3, padding=1),  
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3, padding=1), 
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(400, 512, 3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc1 = nn.Linear(512 * 28 * 28, 1024) 
        if self.datatype=='mathwriting':
            self.fc2 = nn.Linear(1024, 4*self.classes)
        elif self.datatype=='mnist':
            self.fc2 = nn.Linear(1024, self.classes)

    def forward(self, x):
        x = self.conv_block_1(x)  
        x = self.conv_block_2(x) 
        x = self.conv_block_3(x)  
        x = self.conv_block_4(x)  
        x = self.conv_block_5(x) 

        x = x.view(x.size(0), -1)  
        x = self.fc1(x)  
        x = nn.ReLU()(x)  
        x = self.fc2(x)
        if self.datatype=='mathwriting':
            x = x.view(16, 4, 229)
        return x   


class CNN2train(nn.Module):
    def __init__(self, classes, datatype):
        super(CNN2train, self).__init__()
        self.classes = classes
        self.datatype = datatype
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, padding=1),  
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 100, 3, padding=1),  
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(100, 150, 3, padding=1),  
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.Conv2d(150, 200, 3, padding=1),  
            nn.BatchNorm2d(200),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(200, 250, 3, padding=1), 
            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(250, 300, 3, padding=1), 
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(300, 350, 3, padding=1),  
            nn.BatchNorm2d(350),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(350, 400, 3, padding=1),  
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(400, 512, 3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc1 = nn.Linear(512 * 28 * 28, 1024) 
        if self.datatype=='mathwriting':
            self.fc2 = nn.Linear(1024, 4 * self.classes)
        elif self.datatype=='mnist':
            self.fc2 = nn.Linear(1024, self.classes)

    def forward(self, x):
        x = self.conv_block_1(x)  
        x = self.conv_block_2(x) 
        x = self.conv_block_3(x)  
        x = self.conv_block_4(x)  
        x = self.conv_block_5(x) 

        x = x.view(x.size(0), -1)  
        x = self.fc1(x)  
        x = nn.ReLU()(x)  
        x = self.fc2(x)
        if self.datatype=='mathwriting':
            x = x.view(16, 4, 229)
        return x   


class CNN1train(nn.Module):
    def __init__(self, classes, datatype):
        super(CNN1train, self).__init__()
        self.classes = classes
        self.datatype = datatype
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  
            nn.ReLU(),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(),
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(512 * 28 * 28, 1024) 
        if self.datatype=='mathwriting':
            self.fc2 = nn.Linear(1024, 4 * self.classes)
        elif self.datatype=='mnist':
            self.fc2 = nn.Linear(1024, self.classes)

    def forward(self, x):
        x = self.conv_block_1(x)  
        x = self.conv_block_2(x) 
        x = self.conv_block_3(x)  
        x = self.conv_block_4(x)  
        x = self.conv_block_5(x) 

        x = x.view(x.size(0), -1)  
        x = self.fc1(x)  
        x = nn.ReLU()(x)  
        x = self.fc2(x)
        if self.datatype=='mathwriting':
            x = x.view(16, 4, 229)
        return x   
