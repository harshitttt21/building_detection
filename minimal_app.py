from flask import Flask, render_template, jsonify
import joblib
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load data
print("Loading satellite data...")
images = joblib.load('saber_data/ims.np', mmap_mode='r')
masks = joblib.load('saber_data/mas.np', mmap_mode='r')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data_info')
def get_data_info():
    return jsonify({
        'total_images': images.shape[0],
        'image_shape': images.shape[1:],
        'bands': images.shape[-1],
        'mask_shape': masks.shape[1:]
    })

@app.route('/api/image/<int:idx>')
def get_image(idx):
    """Get image data for specific index"""
    if idx >= images.shape[0]:
        return jsonify({'error': 'Image index out of range'}), 400
    
    # Extract data
    img_data = images[idx]
    mask_data = masks[idx]
    
    # Create visualizations
    visualizations = {}
    
    # RGB visualization
    rgb = img_data[:, :, 0:3]
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    visualizations['rgb'] = create_image_base64(rgb_norm)
    
    # Mask visualization
    visualizations['mask'] = create_image_base64(mask_data, cmap='gray')
    
    # Statistics
    stats = {
        'mean_rgb': float(rgb.mean()),
        'building_coverage': float((mask_data > 0).mean() * 100)
    }
    
    return jsonify({
        'index': idx,
        'visualizations': visualizations,
        'stats': stats
    })

def create_image_base64(data, cmap=None):
    """Convert numpy array to base64 encoded image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap=cmap)
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
