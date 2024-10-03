# Fashion Recommendation System

This project implements a Fashion Recommendation System using a pre-trained ResNet50 model for feature extraction from images. The system utilizes a Nearest Neighbors algorithm to recommend similar fashion items based on an uploaded image.

## Features

- Upload an image from your local gallery.
- Extracts features from the uploaded image using ResNet50.
- Recommends similar images from a pre-defined dataset.

## Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Streamlit
- Scikit-learn
- PIL
- TQDM

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-recommendation-system.git
   cd fashion-recommendation-system
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your images in the `images/` directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the URL provided in your terminal to access the application.

## Data Preparation

Before running the application, ensure that you have extracted features from the dataset images using the provided code. The features and filenames will be saved as `features_list.pkl` and `features_name.pkl`.

## Contributing

Feel free to fork the repository and make improvements. Pull requests are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### requirements.txt

```
pandas
numpy
tensorflow
keras
scikit-learn
streamlit
Pillow
tqdm
```


