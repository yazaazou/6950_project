#!/bin/bash

python weekly_gauss_fit.py

python monthly_temp_dist.py

pytest -v unit_tests.py
