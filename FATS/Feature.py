import os
import sys
import time
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import featureFunction


class FeatureSpace:
    """
    This Class is a wrapper class, to allow user select the
    features based on the available time series vectors (magnitude, time,
    error, second magnitude, etc.) or specify a list of features.

    __init__ will take in the list of the available data and featureList.

    User could only specify the available time series vectors, which will
    output all the features that need this data to be calculated.

    User could only specify featureList, which will output
    all the features in the list.

    User could specify a list of the available time series vectors and
    featureList, which will output all the features in the List that
    use the available data.

    Additional parameters are used for individual features.
    Format is featurename = [parameters]

    usage:
    data = np.random.randint(0,10000, 100000000)
    # automean is the featurename and [0,0] is the parameter for the feature
    a = FeatureSpace(category='all', automean=[0,0])
    print a.featureList
    a=a.calculateFeature(data)
    print a.result(method='array')
    print a.result(method='dict')

    """
    def __init__(self, Data=None, featureList=None, excludeList=[], **kwargs):
        self.featureFunc = []
        self.featureList = []
        self.featureOrder = []
        self.featureList = []

        self.sort = False

        if Data is not None:
            self.Data = Data

            if self.Data == 'all':
                if featureList == None:

                    if excludeList == None:
                        for name, obj in inspect.getmembers(featureFunction):
                            if inspect.isclass(obj) and name != 'Base':
                                if obj.__module__.endswith('FeatureFunctionLib'):
                                    # if set(obj().Data).issubset(self.Data):
                                    self.featureOrder.append((inspect.getsourcelines(obj)[-1:])[0])
                                    self.featureList.append(name)
                    else:
                        for name, obj in inspect.getmembers(featureFunction):
                            if inspect.isclass(obj) and name != 'Base' and not name in excludeList:
                                if obj.__module__.endswith('FeatureFunctionLib'):
                                    # if set(obj().Data).issubset(self.Data):
                                    self.featureOrder.append((inspect.getsourcelines(obj)[-1:])[0])
                                    self.featureList.append(name)

                else:
                    for feature in featureList:
                        for name, obj in inspect.getmembers(featureFunction):
                            if name != 'Base':
                                if inspect.isclass(obj) and feature == name:
                                    self.featureList.append(name)

            else:

                if featureList is None:
                    for name, obj in inspect.getmembers(featureFunction):
                        if inspect.isclass(obj) and name != 'Base' and not name in excludeList:
                            if obj.__module__.endswith('FeatureFunctionLib'):
                                if name in kwargs.keys():
                                    if set(obj(kwargs[name]).Data).issubset(self.Data):
                                        self.featureOrder.append((inspect.getsourcelines(obj)[-1:])[0])
                                        self.featureList.append(name)

                                else:
                                    if set(obj().Data).issubset(self.Data):
                                        self.featureOrder.append((inspect.getsourcelines(obj)[-1:])[0])
                                        self.featureList.append(name)
                                    else:
                                        print "Warning: the feature", name, "could not be calculated because", obj().Data, "are needed."
                else:

                    for feature in featureList:
                        for name, obj in inspect.getmembers(featureFunction):
                            if name != 'Base':
                                if inspect.isclass(obj) and feature == name:
                                    if set(obj().Data).issubset(self.Data):
                                        self.featureList.append(name)
                                    else:
                                        print "Warning: the feature", name, "could not be calculated because", obj().Data, "are needed."

            if self.featureOrder != []:
                self.sort = True
                self.featureOrder = np.argsort(self.featureOrder)
                self.featureList = [self.featureList[i] for i in self.featureOrder]
                self.idx = np.argsort(self.featureList)

        else:
            self.featureList = featureList

        m = featureFunction

        for item in self.featureList:
            if item in kwargs.keys():
                try:
                    a = getattr(m, item)(kwargs[item])
                except:
                    print "error in feature " + item
                    sys.exit(1)
            else:
                try:
                    a = getattr(m, item)()
                except:
                    print " could not find feature " + item
                    # discuss -- should we exit?
                    sys.exit(1)
            try:
                self.featureFunc.append(a.fit)
            except:
                print "could not initilize " + item

    def calculateFeature(self, data):
        self._X = np.asarray(data)
        self.__result = []
        for f in self.featureFunc:
            self.__result.append(f(self._X))
        return self

    def result(self, method='array'):
        if method == 'array':
            if self.sort == True:
                return [np.asarray(self.__result)[i] for i in self.idx]
            else:
                return np.asarray(self.__result)
        elif method == 'dict':
            if self.sort == True:
                return dict(zip([self.featureList[i] for i in self.idx], [np.asarray(self.__result)[i] for i in self.idx]))
            else:
                return dict(zip(self.featureList, np.asarray(self.__result)))
        elif method == 'features':
            if self.sort == True:
                return [self.featureList[i] for i in self.idx]
            else:
                return self.featureList
        else:
            return self.__result

