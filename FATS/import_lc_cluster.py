
#from Feature import FeatureSpace
import numpy as np

class ReadLC_MACHO:


    def __init__(self,lc):

        self.content1=lc

    def ReadLC(self):

        data = []
        mjd = []
        error = []
        # Opening the blue band
        #fid = open(self.id,'r')

        self.content1 = self.content1[3:]


        for i in xrange(len(self.content1)):
            if not self.content1[i]:
                break
            else:
                content = self.content1[i].split(' ')
                mjd.append(float(content[0]))
                data.append(float(content[1]))
                error.append(float(content[2]))

        # Opening the red band

        return [data, mjd, error]

    