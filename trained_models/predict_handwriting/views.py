from django.shortcuts import render
# Create your views here.
from django.http import JsonResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from .forms import UploadFileForm
import matplotlib.pyplot as plt 
from PIL import ImageOps
from PIL import Image
from PIL import ImageEnhance 






model = load_model("predict_handwriting\my_model.keras")

#C:\Users\User\Desktop\Python - Practice\BCA-project\bca-project-handwriting\trained_models\predict_handwriting\handwritten_digit_model.h5
print(model.summary())
def index(request):
    form = UploadFileForm()
    return render(request, "index.html", {"form": form})


def predict(request):
    if request.method == "POST":
        image = request.FILES["file"]
        #image = UploadFileForm(request.POST, request.FILES)
        ImgSrc = Image.open(image).convert("L").resize((28, 28))
        img = ImageOps.invert(ImgSrc) 
        
        plt.imshow(img, cmap=plt.cm.binary)
        plt.title("Preprocessed Image Before Prediction")
        plt.axis("off")
        plt.show()

        img = np.array(img).reshape(1, 28, 28) / 255.0

        prediction = model.predict(img)
   
        print(prediction)
        return JsonResponse({"prediction": int(np.argmax(prediction))})