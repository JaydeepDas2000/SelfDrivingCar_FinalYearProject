print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import * # user define module
from sklearn.model_selection import train_test_split # spliting data

# STEP 1 & 2 - Import Data & Trim Data
path = 'myData'
data = importDataInfo(path)

# STEP 3 - Visualization and Distrubation of Data
data = balanceData(data, display = False)

# Step 4 : Preparing for Processing
imagesPath, steerings = loadData(path,data)
print(imagesPath[0], steerings[0])

# Step 5 : Splitting of Data (Training, Validation)
xTrain, xVal, yTrain, yVal  = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5) ## Traning => 80% and Validation => 20%

print('Total Traning images : ', len(xTrain))
print('Total Validation images : ',len(xVal))

# Step 6 : Images augmentation
    ## watch in Utils.py or code_analyse.ipynb

# Step 7 : Pre-processing of Images
    ## watch in utils.py

# Step 8 : Creating model proposed by NVIDIA
model = createModel()
model.summary()

# Step 9 : Training Model
history = model.fit(
    batchGen(xTrain, yTrain,  100, 1),steps_per_epoch=300,epochs=10,
    validation_data=batchGen(xVal, yVal, 100, 0),validation_steps=200
)

# STEP 10 : Save the model
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylabel([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()