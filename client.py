
import os
import torch
import flwr as fl

from torch import nn, optim
from logging import INFO, DEBUG
from flwr.common.logger import log
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets


class CnnModel( nn.Module ) :
    def __init__( self, num_classes = 2 ):
        super( CnnModel, self ).__init__()
        log( DEBUG, 'CnnModel.__init__' )
        
        self.conv1 = nn.Conv2d( in_channels = 3, out_channels = 12, kernel_size = 3, stride = 1, padding = 1 )
        self.bn1   = nn.BatchNorm2d( num_features = 12 )
        self.relu1 = nn.ReLU()   
        self.pool  = nn.MaxPool2d( kernel_size = 2 )
        self.conv2 = nn.Conv2d( in_channels = 12, out_channels = 20, kernel_size = 3, stride = 1, padding = 1 )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d( in_channels = 20, out_channels = 32, kernel_size = 3, stride = 1, padding = 1 )
        self.bn3   = nn.BatchNorm2d( num_features = 32 )
        self.relu3 = nn.ReLU()
        self.fc    = nn.Linear( in_features = 32 * 112 * 112, out_features = num_classes )
        
    def forward( self, input ):
        output = self.conv1( input )
        output = self.bn1( output )
        output = self.relu1( output )
        output = self.pool( output )
        output = self.conv2( output )
        output = self.relu2( output )
        output = self.conv3( output )
        output = self.bn3( output )
        output = self.relu3( output )            
        output = output.view( -1, 32 * 112 * 112 )
        output = self.fc( output )
        return output


def data_transforms( phase = None ):
    log( DEBUG, 'data_transforms ({})'.format( phase ) )
    if phase.startswith( 'train' ):
        data = T.Compose( [
            T.Resize( size = ( 256, 256 ) ),
            T.RandomRotation( degrees = ( -20, +20 ) ),
            T.CenterCrop( size = 224 ),
            T.ToTensor(),
            T.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
        ] )
    
    elif phase.startswith( 'test' ) or phase.startswith( 'val' ):
        data = T.Compose( [
            T.Resize( size = ( 224, 224 ) ),
            T.ToTensor(),
            T.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
        ] )
        
    return data

def load_data( client_id ):
    global data_dir
    TEST = 'test' + '_' + client_id
    TRAIN = 'train' + '_' + client_id
    VAL ='val' + '_' + client_id
    
    trainset = datasets.ImageFolder( os.path.join( data_dir, TRAIN ), transform = data_transforms( TRAIN ) )
    testset  = datasets.ImageFolder( os.path.join( data_dir, TEST ),  transform = data_transforms( TEST ) )
    validset = datasets.ImageFolder( os.path.join( data_dir, VAL ),   transform = data_transforms( VAL ) )
    
    trainloader = DataLoader( trainset, batch_size = 64, shuffle = True )
    validloader = DataLoader( validset, batch_size = 64, shuffle = True )
    testloader  = DataLoader( testset,  batch_size = 64, shuffle = True )

    num_examples = { 'trainset': len( trainset ), 'testset': len( testset ), 'validset': len( validloader ) }
    log( INFO, 'Loaded datasets ( #trainset: {}; #testset: {}, #validset: {})'.format( len( trainset ), len( testset ), len( validset ) ) )
    
    return trainloader, testloader, validloader, num_examples


def train( model, trainloader, epochs = 1 ):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam( model.parameters(), lr = 0.01 )
    losses = []
    for jj in range( epochs ):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model( images )
            loss = criterion( output, labels )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            lloss = running_loss / len( trainloader )
            log( DEBUG, 'trained {} epoch with loss: {}'.format( jj, lloss ) )
            losses.append( lloss )
    return losses


def test( model, testloader ):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model( images )
            loss += criterion( outputs, labels ).item()
            _, predicted = torch.max( outputs.data, 1 )
            total += labels.size( 0 )
            correct += ( predicted == labels ).sum().item()
    accuracy = correct / total
    return loss, accuracy
        

class CnnClient( fl.client.NumPyClient ):
    def get_parameters( self, config ):
        x = [ val.cpu().numpy() for _, val in model.state_dict().items() ]
        log( DEBUG, 'CnnClient: get_parameters ({})'.format( len( x ) ) )
        return x

    def set_parameters( self, parameters ):
        params_dict = zip( model.state_dict().keys(), parameters )
        state_dict = OrderedDict( { k: torch.tensor( v ) for k, v in params_dict } )
        model.load_state_dict(state_dict, strict=True)
        log( DEBUG, 'CnnClient: set_parameters' )

    def fit( self, parameters, config ):
        log( DEBUG, 'CnnClient: starning fit' )
        self.set_parameters( parameters )
        train( model, trainloader, epochs = 1 )
        log( DEBUG, 'CnnClient: ening fit' )
        return self.get_parameters( config = {} ), num_examples[ 'trainset' ], {}

    def evaluate( self, parameters, config ):
        log( DEBUG, 'CnnClient: starting evaluate' )
        self.set_parameters( parameters )
        loss, accuracy = test( model, testloader )
        log( DEBUG, 'CnnClient: ending evaluate (accuracy: {})'.format( accuracy ) )
        return float( loss ), num_examples[ 'testset' ], { 'accuracy': float( accuracy ) }


data_dir = 'chest_xray'
n_client = '1'

DEVICE = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
log( DEBUG, 'main: setting up {}'.format( DEVICE ) )

model = CnnModel().to(DEVICE)
trainloader, testloader, validloader, num_examples = load_data( n_client )

#losses = train( model, trainloader, epochs = 10 )
#for idx, loss in enumerate( losses ):
#    print( 'Epoch {} - Loss: {}'.format( idx, loss ) )
#loss, accuracy = test( model, testloader )
#print( 'Model test accuracy: {}; Model test loss: {}'.format( accuracy, loss ) )

log( DEBUG, 'main: running flwr client {}'.format( n_client ) )
fl.client.start_numpy_client( server_address = "[::]:8080", client = CnnClient() )