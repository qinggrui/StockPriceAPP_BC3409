### LSTM For Stock price prediction

import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from pandas_datareader import data as wb
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler




