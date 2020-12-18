import numpy as np
from impl import MRPTNode

class MRPT:

    def __init__(self, config=None):
        self.initialized = False

        if config is None:
            self.config = {}
        else:
            self.config = config

        self.config["max_depth"] = config.get("max_depth", 3)
        self.config["alpha"] = config.get("alpha", 0.1)
        assert self.config["alpha"] > 0
        self.config["min_split"] = config.get("min_split", 10)


    def initialize(self, data_array):
        assert len(data_array.shape) == 2
        self.config["size"], self.config["dim"] = data_array.shape

        # initialize split vectors
        self.split_vectors = []
        for i in range(self.config["max_depth"]+1):
            while True:
                v = np.asarray([np.random.normal(0,1) if p_a < self.config["alpha"] else 0.0 for p_a in np.random.random(self.config["dim"])])
                if np.count_nonzero(v) > 0:
                    self.split_vectors.append(v)
                    break
        # build tree
        self.root = MRPTNode(data_array, np.arange(self.config["size"]), None, self)

        self.initialized = True

    def getSplitVector(self, depth):
        return self.split_vectors[depth]

    def findNearest(self, data_point):
        if not self.initialized:
            raise Exception("Not initialized!")

        return self.root.findNearest(data_point)

if __name__ == "__main__":
    points = np.asarray([
        (0.5,0.5),
        (0.5,0.6),
        (0.5,0.4),
        (0.5,0.45),
        (0.5,0.48),
        (0.5,0.6),
        (-1.0,0.5),
    ])

    mrpt_test = MRPT(config={"alpha": 0.9, "min_split": 2})
    mrpt_test.initialize(points)

    query_point = np.asarray([-0.9,0.4])
    nearest_point_idx = mrpt_test.findNearest(query_point)
    print("Query point:")
    print(query_point)
    print("Nearest point was:")
    print(nearest_point_idx)
    print(points[nearest_point_idx])

    import time
    # Time for 1M entries

    print("Building huge dataset (1M data points)")
    points = np.random.random((1000000,200))

    print("Building tree...")
    time_build_start = time.time()
    mrpt_test = MRPT(config={"alpha": 0.9, "min_split": 50, "max_depth": 200})
    mrpt_test.initialize(points)

    time_build_stop = time.time()
    print("Building tree took: {}s".format(time_build_stop-time_build_start))

    print("Query point")
    query_point = np.random.random(200)
    time_query_start = time.time()
    nearest_point_idx = mrpt_test.findNearest(query_point)
    time_query_stop = time.time()
    print("Querying point took: {}s".format(time_query_stop-time_query_start))
    print("RESULTS:")
    print("Query point:")
    print(query_point)
    print("Nearest point was:")
    print(nearest_point_idx)
    print(points[nearest_point_idx])