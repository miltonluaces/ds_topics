import sys
sys.path.append('D:/source/repos')
from utilities.std_imports import *
import os 
import seaborn as sns
from PIL import Image 
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import tensorflow.keras.utils as tku
import tensorflow.keras.preprocessing as tkp
import tensorflow.keras.applications as tka
import ds_topics.deep_learning.dl_utilities as du

def load_img(img_path, img_name):
    img_pathname = img_path + img_name
    img = tkp.image.load_img(img_pathname, target_size=(224, 224))
    img = tkp.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tka.vgg16.preprocess_input(img)
    return img 

def predict(model, img_path, img_names):
    f, ax = plt.subplots(1, 4)
    f.set_size_inches(80, 40)
    for i in range(4):
        ax[i].imshow(Image.open(img_path + img_names[i]).resize((200, 200), Image.ANTIALIAS))
    plt.show()
    
    f, axes = plt.subplots(1, 4)
    f.set_size_inches(80, 20)
    for i,img_name in enumerate(img_names):
        img = load_img(img_path, img_name)
        preds  = tka.vgg16.decode_predictions(model.predict(img), top=3)[0]
        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes[i])
        b.tick_params(labelsize=55)
        f.tight_layout()