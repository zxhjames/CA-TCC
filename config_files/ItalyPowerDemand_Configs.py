class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.2 # FCN=0.2 SGRU=0.45 LFCN=0.2
        self.features_len = 5

        # training configs
        self.num_epoch = 5 # FCN=5
        # ItalyPowerDemand （FCN=5,SGRU=4,LFCN=10,Transformer=3)


        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 16 # FCN=16 SGRU=16 LSTM-CNN=16 Transformer=16 CDCNN=16

        # todo baseline setting
        self.hidden_size = 128
        self.num_layers = 3
        self.embed = 24
        self.pad_size = 64
        self.bidirectional = True

        # todo LSTM-FCN
        self.NumClassesOut = self.num_classes
        self.N_time = self.input_channels
        self.N_Features = self.embed
        self.N_LSTM_Out = 128
        self.N_LSTM_layers = 1
        self.Conv1_NF = 128
        self.Conv2_NF = 256
        self.Conv3_NF = 128
        self.lstmDropP = 0.2 # 0.8
        self.FC_DropP = 0.2 # 0.3

        # todo cdil-CNN
        self.cdil_input_dim = self.input_channels
        self.cdil_out_dim = self.num_classes
        self.cdil_hidden_channel = 128
        self.cdil_layers = 4
        self.cdil_kernel_size = 3
        self.cdil_dropout = 0.15
        self.cdil_use_embed = False
        self.cdil_char_vocab = None

        '''
        
        USE_EMBED = False

        SEQ_LENGTH = 200
        INPUT_DIM = 1
        OUTPUT_CLASS = 11
    
        BATCH = 64
        HIDDEN_CHANNEL = 20
        LAYER = 3

        '''

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10  #  FCN=4 CD_CNN=10 SGRU=3 Transformer=2 LFCN=3
