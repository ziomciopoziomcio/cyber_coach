
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
from typing import Optional, Callable
import logging
import requests
import time

