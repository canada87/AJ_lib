import numpy as np

class size_manipulator:
    """
    class to manipulate the array dimensions

    :param data: the original matrix
    """

    def __init__(self, data):
        self.data = data

    def rescape(self, column_selected):
        """
        create a new matrix with the selected columns

        :param column_selected: a vector with the idex numbers of the desired columns

        :return: Y[len(data[:,0]), len(column_selected)]
        """

        data_temp = np.zeros((len(self.data[:,0]), len(column_selected)))

        for i in range(len(column_selected)):
            data_temp[:,i] = self.data[:,column_selected[i]]

        return data_temp

    def cut_a_piece(self, x, x_min, x_max, col):
        """
        replace the data in the selected range with the interpolation between the range coordinates

        :param x: array of the x coordinates
        :param x_min: lower limit of the range
        :param x_max: upper limit of the range
        :param col: vector with the column selected for the cut

        :return: Y[len(data[:,0]), len(data[0,:])]
        """

        data_temp = self.data

        for i in range(len(x)):
            if x[i] <= x_min:
                x0_i = i
            if x[i] <= x_max:
                x1_i = i

        x0 = x[x0_i]
        x1 = x[x1_i]

        for j in range(len(col)):
            y0 = self.data[x0_i,col[j]]
            y1 = self.data[x1_i,col[j]]
            for i in range(x0_i, x1_i):
                data_temp[i,col[j]] = y0 + (y1 - y0)*(x[i] - x0)/(x1 - x0)

        return data_temp

    def nanremoval(self, x, data_time = 'matrix'):
        """
        remove the nan and inf value from the array

        the new vectors has a new length which is always shorter then the original

        :param x: array of the x coordinates
        :return: X[:], Y[:]
        """
        if data_time == 'matrix':
            tracce = []
            for i in range(len(x)):
                for j in range(len(self.data[0,:])):
                    if np.isinf(self.data[i,j]):
                        tracce.append(i)
                        break
                    if np.isnan(self.data[i,j]):
                        tracce.append(i)
                        break

            tracce.append(len(x))

            data_temp = np.zeros((len(self.data[:,0])-len(tracce)+1, len(self.data[0,:])))
            x_temp = np.zeros((len(x)-len(tracce)+1))

            q = 0
            ii = 0
            for i in range(len(x)):
                if i != tracce[q]:
                    x_temp[ii] = x[i]
                    data_temp[ii,:] = self.data[i,:]
                    ii = ii + 1
                else:
                    q = q + 1
        else:
            tracce = []
            for i in range(len(x)):
                if np.isinf(self.data[i]):
                    tracce.append(i)
                if np.isnan(self.data[i]):
                    tracce.append(i)
            # print(self.data[i])

            tracce.append(len(x))

            data_temp = np.zeros(len(self.data[:])-len(tracce)+1)
            x_temp = np.zeros((len(x)-len(tracce)+1))

            q = 0
            ii = 0
            for i in range(len(x)):
                if i != tracce[q]:
                    x_temp[ii] = x[i]
                    data_temp[ii] = self.data[i]
                    ii = ii + 1
                else:
                    q = q + 1

        return x_temp, data_temp

    def nansubstitute(self, x):
        """
        substitute the nan and inf value from the array with an interpolation

        :param x: array of the x coordinates
        :return: Y[len(data)]
        """

        for i in range(len(x)):
            for j in range(len(self.data[0,:])):
                if np.isinf(self.data[i,j]):
                    self.data[i,j] = self.data[i-1,j] + (self.data[i+1,j] - self.data[i-1,j])*(x[i] - x[i-1])/(x[i+1] - x[i-1])

                if np.isnan(self.data[i,j]):
                    self.data[i,j] = self.data[i-1,j] + (self.data[i+1,j] - self.data[i-1,j])*(x[i] - x[i-1])/(x[i+1] - x[i-1])

        return self.data

    def salva_generico(self, num_row, num_col, namefile = 'data.txt'):
        file = open(namefile, 'w')
        for i in range(num_row):
            for j in range(num_col):
                file.write(str(self.data[i][j])+'    ')
            file.write('\n')
        file.close()

    def integrale(self, x, lim_inf, lim_sup):

        iinf = 0
        isup = len(x)-1

        for i in range(len(x)):
            if lim_inf>x[i]:
                iinf = i
            if lim_sup>x[i]:
                isup = i

        print(x[iinf], ' ', x[isup])

        area = 0
        for i in range(iinf, isup):
            x0 = x[i]
            x1 = x[i+1]
            y0 = self.data[i]
            y1 = self.data[i+1]
            area = (x1 - x0)*(y0) + (x1 - x0)*(y1-y0)/2 + area
        return area
