from django import forms


class QuestionForm(forms.Form):
    question = forms.CharField(max_length=200, label="Write a question based on the left text")



