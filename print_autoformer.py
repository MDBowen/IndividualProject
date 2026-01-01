from torchsummary import summary
from Autoformer.exp.exp_main import Exp_Main

from create_parser import return_parser


"""Move into Autoformer to get working"""

if __name__=='__main__':


    parser = return_parser()



    Exp = Exp_Main

    exp = Exp(parser)

    print(exp.model)

