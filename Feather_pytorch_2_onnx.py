import sys
sys.path.insert(0,'.')
sys.path.insert(0,'..')
import torch
from models import FeatherNet
if __name__=='__main__':
    name = 'FeatherNetB'
    net = FeatherNet.mobilelitenetB()
    #print(net)
    model_path = './checkpoints/mobilelitenetB_bs32/_47_best.pth.tar'
    checkpoint = torch.load(model_path,map_location = 'cpu')
    print('load model:',model_path)
    model_dict = {}
    state_dict = net.state_dict()
    #print(checkpoint)
    for (k,v) in checkpoint['state_dict'].items():
        print(k)
        if k[7:] in state_dict:
            model_dict[k[7:]] = v
    state_dict.update(model_dict)
    net.load_state_dict(state_dict)
    #net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    dummy_input = torch.randn([1,3,224,224])
    torch.onnx.export(net,dummy_input,'feathernetB.onnx',verbose=True)
