from django.views.generic import FormView
from .forms import UploadImageForm
import numpy as np
import cv2
import base64
from . import LOGISTIC_REGRESSION_MODEL

def to_data_uri(image):
    data64 = base64.b64encode(image)
    return u'data:img/jpeg;base64,' + data64.decode('utf-8')


class MyFormView(FormView):
    template_name = "form.html"
    form_class = UploadImageForm
    success_url = "success_url"  # Redirect URL on form success

    def form_valid(self, form):
        context = self.get_context_data(form=form)
        image = form.cleaned_data['image'].file.read()
        image_array = cv2.imdecode(np.frombuffer(image, np.uint8), - 1)
        prediction = LOGISTIC_REGRESSION_MODEL.predict(np.array([image_array.flatten()]))[0]
        context['image'] = to_data_uri(image)
        context['result'] = f'This image is an {prediction}'
        return self.render_to_response(context=context)
