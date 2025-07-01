# Système et fichiers
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Fichiers et données
import csv
import json
import itertools
import numpy as np
import pandas as pd
import polars as pl
import fastparquet as fp

# Visualisation
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Optimisation et mathématiques
from scipy.optimize import curve_fit
from random import *
import math

# Progression et outils
from tqdm import tqdm

print('Libraires imported correctly')