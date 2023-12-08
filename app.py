from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
# Change the import statement from this:
# from collections import Sequence

# To this:
from collections.abc import Sequence



from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os



classes = {
    0: ("actinic keratoses and intraepithelial carcinomae(Cancer)"),
    1: ("basal cell carcinoma(Cancer)"),
    2: ("benign keratosis-like lesions(Non-Cancerous)"),
    3: ("dermatofibroma(Non-Cancerous)"),
    4: ("melanocytic nevi(Non-Cancerous)"),
    5: ("pyogenic granulomas and hemorrhage(Can lead to cancer)"),
    6: ("melanoma(Cancer)"),
}
model = load_model('./best_model.h5')







@app.route('/')
def krishna_route():
    return 'Hello, world!'


@app.route('/upload', methods=['POST'])
def upload():

    print("upload\n\n\n\n\n\n\n\n\n")
    if 'image' in request.files:
        image_file = request.files['image']
        print("upload2\n\n\n\n\n\n\n\n\n")



    # Check if the file has an allowed extension
    if image_file:
        # Save the file to the root folder of the server
        file_path = os.path.join('./', "temp.png")
        image_file.save(file_path)
        print("saved\n\n\n\n")








    # # Load the 
    # img = Image.open(image_file)
    # img = img.convert('RGB')

    # # Resize the image to the target size (28, 28)
    # img_resized = img.resize((28, 28))
    # # Convert the resized image to a NumPy array
    # image = img_to_array(img_resized)
    # # Normalize the pixel values to be between 0 and 1 (if needed)
    # image /= 255.0


 


    image = load_img('./temp.png', target_size=(28, 28))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    prediction = model.predict(image)

    # Get the predicted class and label
    predicted_class = np.argmax(prediction[0])
    label = classes[predicted_class]

    # Print the predicted class and label
    print('Predicted class:', predicted_class)
    print('Predicted label:', label)
    # Process or save the image data as needed
    # For example, you can save it to a file or perform further processing

    # Dummy response for demonstration purposes
    response_data = {"label":label}


    try:
        # Check if the file exists before attempting to delete
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(jsonify({'error': f'File not found'}))
    except Exception as e:
        print(jsonify({'error': f'Error deleting file: {str(e)}'}))




    
    return jsonify(response_data)

    # return jsonify({'error': 'No image file provided'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')