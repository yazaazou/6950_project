#!/bin/bash

python weekly_temp_dist.py

python monthly_temp_dist.py

pytest -v unit_tests.py
