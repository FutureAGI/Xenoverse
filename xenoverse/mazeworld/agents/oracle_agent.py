import numpy
import math
import pygame
from .smart_slam_agent import SmartSLAMAgent

class OracleAgent(SmartSLAMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._long_term_memory = numpy.ones_like(self._long_term_memory)