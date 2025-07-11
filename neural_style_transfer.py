import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os

class NeuralStyleTransfer:
    def __init__(self, content_layers=['block4_conv2'], 
                 style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 
                              'block4_conv1', 'block5_conv1']):
        """
        Initialize Neural Style Transfer model
        
        Args:
            content_layers: Layers to extract content features
            style_layers: Layers to extract style features
        """
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        
        # Load pre-trained VGG19 model
        self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        
        # Create feature extraction model
        self.extractor = self._create_extractor()
    
    def _create_extractor(self):
        """Create model to extract style and content features"""
        outputs = [self.vgg.get_layer(name).output for name in self.style_layers + self.content_layers]
        model = tf.keras.Model([self.vgg.input], outputs)
        return model
    
    def preprocess_image(self, image_path, max_dim=512):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file or URL
            max_dim: Maximum dimension for resizing
        """
        # Load image
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = np.array(img)
        shape = img.shape[:2]
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tuple([int(shape[i] * scale) for i in range(2)])
        img = tf.image.resize(img, new_shape)
        
        # Normalize pixel values
        img = img[tf.newaxis, :]
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    def deprocess_image(self, processed_img):
        """Convert processed image back to displayable format"""
        x = processed_img * 255
        x = np.array(x, dtype=np.uint8)
        if np.ndim(x) == 4:
            x = np.squeeze(x, axis=0)
        return x
    
    def gram_matrix(self, input_tensor):
        """Calculate Gram matrix for style representation"""
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations
    
    def extract_features(self, image):
        """Extract style and content features from image"""
        preprocessed = tf.keras.applications.vgg19.preprocess_input(image * 255)
        outputs = self.extractor(preprocessed)
        
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]
        
        style_features = [self.gram_matrix(style_output) for style_output in style_outputs]
        content_features = {name: value for name, value in zip(self.content_layers, content_outputs)}
        
        return {'content': content_features, 'style': style_features}
    
    def style_content_loss(self, outputs, style_targets, content_targets, 
                          style_weight=1e-2, content_weight=1e4):
        """Calculate combined style and content loss"""
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        
        # Style loss
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[i] - style_targets[i])**2) 
                              for i in range(len(style_outputs))])
        style_loss *= style_weight / self.num_style_layers
        
        # Content loss
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= content_weight / self.num_content_layers
        
        return style_loss + content_loss
    
    @tf.function
    def train_step(self, image, style_targets, content_targets, optimizer):
        """Single training step"""
        with tf.GradientTape() as tape:
            outputs = self.extract_features(image)
            loss = self.style_content_loss(outputs, style_targets, content_targets)
        
        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
        return loss
    
    def stylize_image(self, content_path, style_path, epochs=10, steps_per_epoch=100):
        """
        Apply style transfer to content image
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            epochs: Number of training epochs
            steps_per_epoch: Steps per epoch
        """
        # Load and preprocess images
        content_image = self.preprocess_image(content_path)
        style_image = self.preprocess_image(style_path)
        
        # Extract target features
        style_targets = self.extract_features(style_image)['style']
        content_targets = self.extract_features(content_image)['content']
        
        # Initialize generated image
        image = tf.Variable(content_image)
        
        # Setup optimizer
        optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        
        # Training loop
        print("Starting style transfer...")
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                loss = self.train_step(image, style_targets, content_targets, optimizer)
                
                if step % 20 == 0:
                    print(f"Epoch {epoch+1}, Step {step}: Loss = {loss:.4f}")
            
            # Display progress
            if (epoch + 1) % 2 == 0:
                self.display_image(image, title=f"Epoch {epoch+1}")
        
        return image
    
    def display_image(self, image, title="Image"):
        """Display processed image"""
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(self.deprocess_image(image))
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def save_image(self, image, filename):
        """Save processed image to file"""
        processed = self.deprocess_image(image)
        img = Image.fromarray(processed)
        img.save(filename)
        print(f"Image saved as {filename}")

# Example usage and demonstration
def main():
    # Initialize style transfer model
    nst = NeuralStyleTransfer()
    
    # Example 1: Van Gogh style
    print("=== EXAMPLE 1: Van Gogh Style Transfer ===")
    print("Content: Landscape photo")
    print("Style: Van Gogh's Starry Night")
    print("Expected result: Landscape with swirling, expressive brushstrokes")
    
    # Example 2: Picasso style
    print("\n=== EXAMPLE 2: Cubist Style Transfer ===")
    print("Content: Portrait photo")
    print("Style: Picasso cubist painting")
    print("Expected result: Portrait with geometric, fragmented appearance")
    
    # Example 3: Monet style
    print("\n=== EXAMPLE 3: Impressionist Style Transfer ===")
    print("Content: Garden photo")
    print("Style: Monet's Water Lilies")
    print("Expected result: Garden with soft, impressionistic brushwork")
    
    # Usage instructions
    print("\n=== USAGE INSTRUCTIONS ===")
    print("1. Install required packages:")
    print("   pip install tensorflow pillow matplotlib requests")
    print("\n2. Use the model:")
    print("   nst = NeuralStyleTransfer()")
    print("   result = nst.stylize_image('content.jpg', 'style.jpg')")
    print("   nst.save_image(result, 'stylized_output.jpg')")
    
    # Advanced usage example
    print("\n=== ADVANCED USAGE EXAMPLE ===")
    advanced_example = """
# Custom layer configuration for different effects
nst_detailed = NeuralStyleTransfer(
    content_layers=['block4_conv2', 'block5_conv2'],  # More content detail
    style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 
                  'block4_conv1', 'block5_conv1', 'block6_conv1']  # More style layers
)

# Batch processing multiple images
content_images = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
style_images = ['style1.jpg', 'style2.jpg']

for i, content in enumerate(content_images):
    for j, style in enumerate(style_images):
        result = nst.stylize_image(content, style, epochs=15)
        nst.save_image(result, f'stylized_{i}_{j}.jpg')
"""
    print(advanced_example)

if __name__ == "__main__":
    main()
