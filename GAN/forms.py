from typing import Text
from django import forms

class Textform(forms.Form):
    Text = forms.CharField(label='description', max_length=100)