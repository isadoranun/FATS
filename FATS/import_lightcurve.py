
#from Feature import FeatureSpace
import numpy as np

class ReadLC_MACHO:


    def __init__(self,id):

        self.id=id

    def ReadLC(self):

        # Opening the blue band
        fid = open(self.id,'r')

        saltos_linea = 3
        delimiter = ' '
        for i in range(0,saltos_linea):
            fid.next()
        LC = []

        for lines in fid:
            str_line = lines.strip().split()
            floats = map(float, str_line)
            #numbers = (number for number in str_line.split())
            LC.append(floats)

        LC = np.asarray(LC)

        data  = LC[:,1]
        error = LC[:,2]
        mjd = LC[:,0]

        # Opening the red band

        return [data, mjd, error]

    