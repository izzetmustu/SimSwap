from .base_options import BaseOptions

class RecognizeOptionsDataset(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument("--Arc_path", type=str, default='./arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--output_path", type=str, default='/mnt/storage1/izzet/datasets/output/CharadesEgo_v1', help="results path")
        self.parser.add_argument('--use_mask', action='store_true', help='Use mask for better result')
        self.parser.add_argument('--crop_size', type=int, default=224, help='Crop of size of input image')
        self.parser.add_argument('--video_path', type=str, default="/mnt/storage1/izzet/datasets/CharadesEgo_v1")
        
        self.isTrain = False