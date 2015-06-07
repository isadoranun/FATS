import numpy as np


def Align_LC(mjd, mjd2, data, data2, error, error2):

    if len(data2) > len(data):

        new_data2 = []
        new_error2 = []
        new_mjd2 = []
        new_mjd = np.copy(mjd)
        new_error = np.copy(error)
        new_data = np.copy(data)
        count = 0

        for index in xrange(len(data)):

            where = np.where(mjd2 == mjd[index])

            if np.array_equal(where[0], []) is False:

                new_data2.append(data2[where])
                new_error2.append(error2[where])
                new_mjd2.append(mjd2[where])
            else:
                new_mjd = np.delete(new_mjd, index - count)
                new_error = np.delete(new_error, index - count)
                new_data = np.delete(new_data, index - count)
                count = count + 1

        new_data2 = np.asarray(new_data2).flatten()
        new_error2 = np.asarray(new_error2).flatten()


    else:

        new_data = []
        new_error = []
        new_mjd = []
        new_mjd2 = np.copy(mjd2)
        new_error2 = np.copy(error2)
        new_data2 = np.copy(data2)
        count = 0
        for index in xrange(len(data2)):
            where = np.where(mjd == mjd2[index])

            if np.array_equal(where[0], []) is False:
                new_data.append(data[where])
                new_error.append(error[where])
                new_mjd.append(mjd[where])
            else:
                new_mjd2 = np.delete(new_mjd2, (index - count))
                new_error2 = np.delete(new_error2, (index - count))
                new_data2 = np.delete(new_data2, (index - count))
                count = count + 1

        new_data = np.asarray(new_data).flatten()
        new_mjd = np.asarray(new_mjd).flatten()
        new_error =  np.asarray(new_error).flatten()

    return new_data, new_data2, new_mjd, new_error, new_error2
    #return new_mjd, new_data, new_error, new_mjd2, new_data2, new_error2
