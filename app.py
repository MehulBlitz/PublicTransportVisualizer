# AI Transit Demand & Overcrowding Intelligence Platform (Render-ready)

import os
import re
import io
import math
import tempfile
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

print("Dependencies successfully imported!")

class TransitDataEngine:
    """Manages dataset ingestion, automatic mapping, cleaning, and feature engineering."""
    def __init__(self):
        self.df_raw = None
        self.df_cleaned = None
        self.df_featured = None
        self.mapping = {}
        self.quality_report = {}
        self.is_demo = False
        self.warnings = []

    @staticmethod
    def generate_demo_dataset(num_days=30):
        """Generates a highly realistic transit dataset with logical peaks, routes, and capacities."""
        routes = ["Route 101 (Metro-Link)", "Route 205 (Downtown Express)", "Route 42 (Suburban Shuttle)"]
        stations = {
            "Route 101 (Metro-Link)": ["Central Station", "Tech Park", "West Terminal"],
            "Route 205 (Downtown Express)": ["North Gate", "Financial District", "Central Station"],
            "Route 42 (Suburban Shuttle)": ["Suburban Mall", "Green Valley", "Central Station"]
        }
        modes = {
            "Route 101 (Metro-Link)": "Train",
            "Route 205 (Downtown Express)": "Bus",
            "Route 42 (Suburban Shuttle)": "Bus"
        }
        capacities = {
            "Route 101 (Metro-Link)": 500,
