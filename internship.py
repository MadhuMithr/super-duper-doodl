import numpy as np
from sklearn.svm import SVC
import os
from tqdm import tqdm
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import threading

dataset_path = 'C:\\Users\\MADHU MITHRA\\Downloads\\dogs-vs-cats\\train\\train'
images_path = []
labels = []

# Gather image paths and labels
for imgp in os.listdir(dataset_path):
    images_path.append(os.path.join(dataset_path, imgp))
    if 'cat' in imgp.lower():
        labels.append(0)
    elif 'dog' in imgp.lower():
        labels.append(1)

# Load images and resize them
img = []
imgsize = (64, 64)
for i in tqdm(images_path):
    im = cv2.imread(i)
    im = cv2.resize(im, imgsize)
    img.append(im)

q = np.array(img)
v = np.array(labels)

# Reduce dataset size (e.g., use 5000 samples for efficiency)
subset_size = 5000  # Adjust size as needed

# Split the data to get a smaller subset randomly
q1, _, v1, _ = train_test_split(q, v, train_size=subset_size, stratify=v, random_state=42)

# Flatten images
q1 = q1.reshape(q1.shape[0], -1)
print("Reshape completed")

# Normalize
scaler = StandardScaler()
q1 = scaler.fit_transform(q1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(q1, v1, test_size=0.2, random_state=42)

# Initialize and train the model
model = SVC()
model.fit(x_train, y_train)
d = model.score(x_test, y_test)
print(f'Model accuracy: {d}')
# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to the Dog vs Cat Classifier API!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    img_resized = cv2.resize(img, imgsize)
    img_flattened = img_resized.flatten().reshape(1, -1)
    img_normalized = scaler.transform(img_flattened)

    # Make prediction
    prediction = model.predict(img_normalized)
    result = 'dog' if prediction[0] == 1 else 'cat'

    return jsonify({'prediction': result})


# Run Flask app in a separate thread
def run_flask_app():
    app.run(debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()
