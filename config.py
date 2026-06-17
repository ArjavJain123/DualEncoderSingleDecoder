import torch

class InferenceConfig:
    img_size = 640
    hidden_dim = 256
    num_heads = 8
    num_decoder_layers = 3
    freeze_text_encoder = True
    
    # Paths
    weights_path = "weights/model640after20ep.pth"
    anno_dir = "annotations"
    output_dir = "demo_outputs"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
    print("InferenceConfig:")
    for attr, value in vars(InferenceConfig).items():
        if not attr.startswith("__") and not callable(value):
            print(f"  {attr}: {value}")