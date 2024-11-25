import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn.functional as F
def load_model(checkpoint_path, device):
    # Load the trained model
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()  # Set model to evaluation mode
    return model

def preprocess_image(image_path, input_size):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(input_size),  # Resize to model's expected input size
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean/std
                             std=[0.229, 0.224, 0.225])
    ])
    # Transform the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

def postprocess_output(output, original_image, input_tensor, output_path):
    output = torch.argmax(output, dim=1).squeeze(0).cpu()  # Predicted class per pixel

    # Convert prediction to a one-hot encoded mask for visualization
    one_hot_prediction = F.one_hot(output, num_classes=3).float()  # Assuming 3 classes
    mask_visual = one_hot_prediction.numpy()  # Convert to NumPy for visualization

    # Create a Matplotlib figure
    fig, arr = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns for original and prediction
    
    # Display the original image
    arr[0].imshow(original_image)
    arr[0].set_title('Original Image')
    arr[0].axis('off')  # Hide axes for clarity
    
    # Display the prediction
    arr[1].imshow(mask_visual, interpolation='nearest')  # Visualize the segmentation mask
    arr[1].set_title('Predicted Segmentation')
    arr[1].axis('off')  # Hide axes for clarity
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure saved at {output_path}")
    plt.close(fig) 



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    )
    model.to(device)
    # Load the model
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'model.pth')  # Update with your checkpoint path
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    # Preprocess the input image
    input_size = (224, 224)  # Adjust as per your model's input requirement
    image_tensor, original_image = preprocess_image(args.image_path, input_size)
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)  # Run the model on input tensor

    # Define output path
    output_path = os.path.splitext(args.image_path)[0] + "_visualization.png"

    # Postprocess the output and save the grid visualization
    postprocess_output(output, original_image, image_tensor, output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args)
