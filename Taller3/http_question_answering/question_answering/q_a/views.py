from django.views.generic import FormView
from .forms import QuestionForm
from . import EMBEDDING_MODEL, TEXT_BASE
from .question_answering import get_most_related_paragraphs


class MyFormView(FormView):
    template_name = "form.html"
    form_class = QuestionForm

    def form_valid(self, form):
        context = self.get_context_data(form=form)
        question = form.cleaned_data['question']
        context['question'] = question
        most_related_paragraphs = get_most_related_paragraphs(EMBEDDING_MODEL, TEXT_BASE, question)
        context['most_related'] = most_related_paragraphs
        context['text'] = TEXT_BASE
        return self.render_to_response(context=context)

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['text'] = TEXT_BASE
        return self.render_to_response(context)
