import numpy as np

class MRPTNode:
    def __init__(self, data, data_ref, parent, ctl):
        self.config = ctl.config
        self.ctl = ctl
        self.parent = parent
        self.depth = self.__determineDepth__()

        # split parameters
        self.child_left = None
        self.child_right = None
        self.split_vector = ctl.getSplitVector(self.depth)
        self.split_median = None

        self.initialize(data, data_ref)

    def initialize(self, data, data_ref):
        assert len(data) > 0

        if len(data) < self.config["min_split"] or self.depth == self.config["max_depth"]:
            # no split anymore
            self.data = data
            self.data_ref = data_ref
        else:
            # project data points on split vector
            projections_len = [ np.linalg.norm(self.__projectOnVector__(data_point)) for data_point in data ]
            # find pivot element
            self.split_median = np.median(projections_len)

            # allocate new data arrays
            n_dataLeft = np.count_nonzero(projections_len < self.split_median)
            n_dataRight = len(data) - n_dataLeft


            if n_dataLeft > 0 and n_dataRight > 0:
                # we can split the node!
                # assemble child data
                child_left_data = np.empty((n_dataLeft, self.config["dim"]), dtype=data.dtype)
                child_left_data_ref = np.empty(n_dataLeft, dtype=data_ref.dtype)
                child_left_ctr = 0

                child_right_data = np.empty((n_dataRight, self.config["dim"]), dtype=data.dtype)
                child_right_data_ref = np.empty(n_dataRight, dtype=data_ref.dtype)
                child_right_ctr = 0
                for i, pivot in enumerate(projections_len):
                    if pivot < self.split_median:
                        child_left_data[child_left_ctr,:] = data[i,:]
                        child_left_data_ref[child_left_ctr] = data_ref[i]
                        child_left_ctr += 1
                    else:
                        child_right_data[child_right_ctr,:] = data[i,:]
                        child_right_data_ref[child_right_ctr] = data_ref[i]
                        child_right_ctr += 1

                self.child_left = MRPTNode(child_left_data, child_left_data_ref, self, self.ctl)
                self.child_right = MRPTNode(child_right_data, child_right_data_ref, self, self.ctl)
            else:
                # we need to stay linear
                self.data = data
                self.data_ref = data_ref


    def isLeaf(self):
        assert (self.child_left is None) == (self.child_right is None)
        return self.child_left is None

    def findNearest(self, data_point):
        if self.isLeaf():
            # search in data, linearly
            matches = []

            for i, stored_data_point in enumerate(self.data):
                matches.append(
                    (i, np.linalg.norm(stored_data_point - data_point))
                )

            matches.sort(key=lambda element: element[1])
            best_idx, best_dist = matches[0]
            return self.data_ref[best_idx]
        else:
            projected = self.__projectOnVector__(data_point)
            projected_len = np.linalg.norm(projected)
            if projected_len < self.split_median:
                return self.child_left.findNearest(data_point)
            else:
                return self.child_right.findNearest(data_point)




    def __projectOnVector__(self, data_point):
        assert len(data_point) == self.config["dim"]

        a = np.zeros(self.config["dim"],dtype=self.split_vector.dtype)
        b = self.split_vector

        projected = a + np.dot(data_point-a,b-a) / np.dot(b-a,b-a) * (b-a)

        return projected

    def __determineDepth__(self):
        depth = 0
        tmp_parent = self.parent
        while tmp_parent is not None:
            depth += 1
            tmp_parent = tmp_parent.parent
        return depth

