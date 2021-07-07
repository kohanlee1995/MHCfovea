from django.shortcuts import render
from django.views import View
from django import forms

import pickle
import numpy as np
from .utils import Interpretation, get_index_image

class SearchForm(forms.Form):
    """form for search
    """
    term = forms.CharField(max_length=50, initial='', required=False)

class Index(View):
    """index page view
    """
    def get(self, request):
        """handle get request
        """

        with open('static/main.pkl', 'rb') as handle:
            pickle_data = pickle.load(handle)

        overview, application = get_index_image()

        data = {
            'allele_list': [key for key in pickle_data['seq']],
            'overview': overview,
            'application': application
        }

        return render(request, 'index.html', data)

class Search(View):
    """search page view
    """
    def get(self, request):
        """handle get request
        """
        form = SearchForm(request.GET)
        if not form.is_valid():
            return render(request, 'error.html', {'error_msg': 'Invalid input'})

        interpretation = Interpretation('static/main.pkl')
        search_term = form.cleaned_data['term']

        if search_term not in interpretation.mhc_dict:
            return render(request, 'error.html', {'error_msg': 'Invalid input'})

        overview_image = interpretation.get_overview_image(search_term)
        detail_image = interpretation.get_detail_image(search_term)

        data = {
            'allele': search_term,
            'overview_image': overview_image,
            'detail_image': detail_image
        }

        return render(request, 'search.html', data)