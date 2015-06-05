import os,sys,time
import numpy as np
class Base:
    def __init__(self):
	self.category='all'
    def fit(self, data):
        return self
