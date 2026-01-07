from exp.exp_main import Exp_Main

from create_parser import return_parser

if __name__=='__main__':


    parser = return_parser()

    Exp = Exp_Main

    exp = Exp(parser)

    print(exp.model)

