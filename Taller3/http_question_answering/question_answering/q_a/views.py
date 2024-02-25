from django.views.generic import FormView
from .forms import QuestionForm
from . import EMBEDDING_MODEL, TEXT_BASE
from .question_answering import get_most_related_paragraphs


class MyFormView(FormView):
    template_name = "form.html"
    form_class = QuestionForm
    success_url = "success_url"  # Redirect URL on form success

    def form_valid(self, form):
        context = self.get_context_data(form=form)
        question = form.cleaned_data['question']
        context['question'] = f'Your question: {question}'
        top_3_related_paragraphs = get_most_related_paragraphs(EMBEDDING_MODEL, TEXT_BASE, question)
        context['more'] = f'Your question: {question}'
        return self.render_to_response(context=context)
