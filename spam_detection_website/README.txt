SPAM DETECTION WEBSITE - RUN INSTRUCTIONS

1. Install Python 3.
2. Open this project folder in VS Code.
3. Open the VS Code terminal in this folder.
4. Install the required packages:
       pip install -r requirements.txt
5. Train the machine-learning model:
       python model.py
   This creates spam_model.pkl.
6. Start the Flask website:
       python app.py
7. Open this address in your browser:
       http://127.0.0.1:5000

PROJECT FILES
- spam.csv: small sample dataset with category and message columns.
- requirements.txt: Python packages required by the project.
- model.py: trains and saves the text-classification model.
- app.py: Flask server and prediction route.
- templates/index.html: website page.
- static/style.css: website design.

NOTE
The included CSV is a small teaching dataset, not a production-quality dataset.
For better results, train with a larger, balanced, correctly labelled dataset.
