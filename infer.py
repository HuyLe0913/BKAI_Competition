import argparse
import torch
from torchvision import transforms
from PIL import Image
import os

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

def postprocess_output(output, original_image):
    # Convert the output tensor to a PIL image
    output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Get class predictions
    output_image = Image.fromarray((output * 255).astype('uint8'))  # Scale for visualization
    output_image = output_image.resize(original_image.size)  # Resize to original image size
    return output_image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    checkpoint_path = 'model.pth'  # Update with your checkpoint path
    model = load_model(checkpoint_path, device)
    
    # Preprocess the input image
    input_size = (224, 224)  # Adjust as per your model's input requirement
    image_tensor, original_image = preprocess_image(args.image_path, input_size)
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Postprocess the output
    segmented_image = postprocess_output(output, original_image)
    
    # Save the result
    output_path = os.path.splitext(args.image_path)[0] + "_segmented.png"
    segmented_image.save(output_path)
    print(f"Segmented image saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args)
