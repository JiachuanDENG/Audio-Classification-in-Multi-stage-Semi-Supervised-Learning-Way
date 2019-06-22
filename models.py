import torch
import torch.nn as nn
import torch.autograd as autograd
class CNNModel(nn.Module):

    def __init__(self,num_classes):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,16,(5,5))
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv1_activation = torch.nn.ReLU()

        self.conv2 = nn.Conv2d(16,32,(3,3))
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_activation = torch.nn.ReLU()

        self.conv3 = nn.Conv2d(32,64,(3,3),stride=2)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv3_activation = torch.nn.ReLU()

        self.LSTM_stack = nn.LSTM(1024,512, num_layers=2, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        # self.FC1 = nn.Linear(64*146*16,256)
        self.FC1 = nn.Linear(512,256)
        self.FC2 = nn.Linear(256,self.num_classes)


    def forward(self, x):
        conv1 = self.conv1_activation(self.conv1_bn(self.conv1(x)))
        conv2 = self.conv2_activation(self.conv2_bn(self.conv2(conv1)))
        conv3 = self.conv3_activation(self.conv3_bn(self.conv3(conv2)))
#         print (conv3.shape)
        # fc_input = conv3.view(-1,64*146*16)
        conv_out = conv3.permute(0,2,1,3).contiguous()
#         print (conv_out.shape)
        lstm_input = conv_out.view(-1,conv_out.size(1), conv_out.size(2)*conv_out.size(3))
#         print (lstm_input.shape)
        lstm_out,_ = self.LSTM_stack(lstm_input)
#         print (lstm_out.shape)
        fc_input = lstm_out[:,-1,:]
# #         print (fc_input.shape)
        fc1 = torch.nn.functional.relu(self.FC1(fc_input))
        out = torch.nn.functional.sigmoid(self.FC2(fc1))
        return out


        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.nn.functional.avg_pool2d(x, 2)
        return x

def conv3x3(in_channels,out_channels,stride=1,groups=1,dilation=1):
    """with padding"""
    return nn.Conv2d(in_channels,out_channels,\
                    kernel_size = 3, stride = stride,\
                     bias = False, padding = dilation,\
                     groups = groups, dilation = dilation)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class ResnetCNNBlock(nn.Module):
    """
    since network will not be very deep,
    do not use bottleneck version
    """
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResnetCNNBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels  = out_channels
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or self.in_channels!=self.out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self,x):
        # print (x.shape)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print (out.shape)
        if self.downsample is not None :
            identity = self.downsample(identity)
        # print ('++++++++++++',out.shape,identity.shape)
        out += identity
        out = self.relu(out)
        return out


class CNNModelv2(nn.Module):
    def __init__(self, num_classes):
        super(CNNModelv2,self).__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
        )



        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        # print (x.shape) # [x,512,8,8]
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x

class CNNModelv3(nn.Module):
    def __init__(self, num_classes):
        super(CNNModelv3,self).__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            ConvBlock(in_channels=128, out_channels=64),
        )

        self.LSTM_stack = nn.GRU(512,256, num_layers=2, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        # print (x.shape) # [x,64 c ,8 t,8 f]
        conv_out = x.permute(0,2,1,3).contiguous() # [x,t,c,f] [x,8,64,8]
        lstm_input = conv_out.view(-1,conv_out.size(1), conv_out.size(2)*conv_out.size(3))#[x,8, 512]
        lstm_out,_ = self.LSTM_stack(lstm_input) # [x,8,256]
        fc_input = lstm_out[:,-1,:]
        fc_out = self.fc(fc_input)
        return fc_out
class CNNModelv4(nn.Module):
    """
    resnet cnn
    """
    def __init__(self,num_classes):
        super(CNNModelv4,self).__init__()
        self.conv = nn.Sequential(
            ResnetCNNBlock(in_channels=3,out_channels=64),
            ResnetCNNBlock(in_channels=64,out_channels=128,stride=2),
            ResnetCNNBlock(in_channels=128,out_channels=256,stride=2),
            ResnetCNNBlock(in_channels=256,out_channels=512,stride=2)

        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
    def forward(self,x):
        out = self.conv(x)
        # print (out.size())
        out = self.avgpool(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out
