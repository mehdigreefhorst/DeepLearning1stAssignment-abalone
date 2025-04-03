# DeepLearning1stAssignment-abalone
 Project for JADS class of deeplearning. The code is distrubted  between three files. data_load.py includes all the code for loading the data into the AbaloneDataset class. In Assignment-1-final.ipynb, the code and instructions can be found for training the model on the data. inference.ipynb includes the code for inference of the best model. The code loads in the model from a .pth file.


### Project setup
- Install a virtual environment (we used python 3.10.12)
    - `python -m venv venv`
    - `pip install -r requirements.txt`
- create a folder ("data") and add the test.csv and train.csv in this folder
- Add the model with name "best_model.pth" in the same folder as the inference notebook
- Run the notebook inference.ipynb (this code is a subset of the code in the notebook Assignment-1-final.ipynb) You can also choose to run the complete Assignment-1-final.ipynb, where you'll train your own model and perform inference in the end. But as you only wanted to do inference, you can simply run the inference.ipynb.

### Final project structure
- data
    - train.csv
    - test.csv
    - final.csv (contains the predicted rings of the test.csv)
- best_model.pth
- data_load.py
- inference.ipynb
- Assignment-1-final.ipynb
- requirements.txt


### Train model yourself
In the Assignment-1-final.ipynb, you can follow all of the code that we implemented to train your own model using the Abalone dataset.
