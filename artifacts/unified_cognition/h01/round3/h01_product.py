"""R2 diagnostic: a symmetric multiplicative readout, same for both core types."""
import argparse
from torch import nn
import torch
import h01
BaseNet=h01.Net
class ProductHead(nn.Module):
    def __init__(self,width):
        super().__init__();self.a=nn.Linear(width,32);self.b=nn.Linear(width,32);self.out=nn.Linear(32,2)
    def forward(self,h):return self.out(torch.tanh(self.a(h))*torch.tanh(self.b(h)))
class ProductNet(BaseNet):
    def __init__(self,kind='fused',width=64):
        super().__init__(kind,width);self.head=ProductHead(width)
h01.Net=ProductNet
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--round',default='round2');p.add_argument('--model',default='F64');p.add_argument('--seed',type=int,default=270201);p.add_argument('--lr',type=float,default=.003);p.add_argument('--steps',type=int,default=2000);p.add_argument('--eval-n',type=int,default=4096);args=p.parse_args()
 h01.run(args)
 path=h01.BASE/args.round/f'{args.model}_s{args.seed}_lr{args.lr:g}'/'result.json'
 import json
 r=json.loads(path.read_text());r['head']='symmetric_product';r['product_source_sha256']=h01.digest(__file__);h01.dump(path,r)
