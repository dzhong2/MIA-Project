import numpy as np
from scipy.stats import entropy


class node:
    def __init__(self, max_depth=None, count_noise=False):
        # initialize attribute index, -1 if it's a root
        self.attrindex = -1
        # initialize attribute value, None if it's a root
        self.attrvalue = None
        # initialize child, None if it's a leaf
        self.child = None
        # initialize probability result, None if it's not a leaf
        self.prob = None
        self.max_depth = max_depth
        self.text = ""
        self.noise = np.array([0, 0])

    def create_child(self, attr_ind, att_value, subdata, sublist, dp, epsilon, noise = np.array([0, 0])):
        '''
        Define a function to create a non-leaf child. Record current attribute index and value and build subtree
        using subdata. The remaining attribute index should be pass into the function so that the correct index will
        be write into the following children. Prob will be left None since this will not create leaf node
        '''
        # Initialize a child node, add attribute index and value. Then add a sub-tree to this child.
        child_node = node(max_depth=self.max_depth-1)
        child_node.attrindex = attr_ind
        child_node.attrvalue = att_value
        child_node.noise = noise
        child_node.build_tree_from_subdata(subdata, sublist, dp, epsilon)
        if self.child is None:
            self.child = []
        self.child.append(child_node)
        #print("Create child")

    def create_leaf(self, attr_ind, att_value, prob, noise=np.array([0, 0])):
        '''
        Define a function to create a leaf child. Record current attribute index and value, record the probability of
        leaf node. Child will be left None.
        '''
        leaf_node = node()
        leaf_node.attrindex = attr_ind
        leaf_node.attrvalue = att_value
        leaf_node.prob = prob
        leaf_node.noise = np.abs(noise)
        if self.child is None:
            self.child = []
        self.child.append(leaf_node)
        #print("Create leaf node")

    def prob_test(self, x):
        '''

        :param x: One record
        :return: The predict probability of this record
        '''
        if self.prob is None:
            '''mean_prob = np.array([0, 0])
            for c in self.child:
                if c.attrvalue == x[c.attrindex]:
                    return c.prob_test(x)
            for c in self.child:
                mean_prob = mean_prob + np.array(c.prob_test(x)) / len(self.child)
            return mean_prob'''
            for c in self.child:
                if c.attrvalue == x[c.attrindex]:
                    return c.prob_test(x)
            return self.child[0].prob_test(x)

        else:
            return self.prob

    def build_tree_from_subdata(self, data, ind_list, dp, epsilon):
        '''
        Build a sub tree basing on data(sub-data) and index_list(considering some index might be removed). The root or
        the parent node should be initialized before running this function. This function follows steps:
        1. If there is more than one attributes left, then:
            a. Calculate the entropy of each attribute and find the one with highest entropy. This attribute is the most
            informative one
            b. Split the data according to the value of the most informative attribute into groups of sub-data.
            It is important to record the true index and update the remaining index.
            c. Create a non-leaf child(a sub tree) using each sub-data.
        2. If there is only one attribute, create a leaf node as following steps:
            a. Split the data according to the only attribute.
            b. Calculate the probability of each class( only 0 or 1 in this project) for each possible value of this
            attribute.
            c. Create a leaf child using the probability of each attribute value.
        '''
        HS = entropy(np.unique(data[:, -1], return_counts=True, axis=0)[1] / len(data))
        if data.shape[1] > 2 and len(data) > 1 and HS > 0 and self.max_depth > 1:
            #Information gain: IG(S,A) = H(S)-H(S|A)
            num_att = data.shape[1]-self.max_depth
            EntList = []
            HSA = []
            for a in range(data.shape[1]- 1):
                paHa = 0
                a_set, counta = np.unique(data[:, a], axis=0, return_counts=True)
                noises = np.random.laplace(0, 2 * data.shape[1] / epsilon, len(a_set))
                counta = np.abs(counta+noises*dp)
                pa = counta/sum(counta)
                #count_a_list = []
                for i in range(len(a_set)):
                    # H(S|A) = sum(p(a)H(a))
                    count_avc = np.array([sum(data[data[:, a] == a_set[i], -1] == 0),
                                          sum(data[data[:, a] == a_set[i], -1] == 1)])
                    #count_avc = abs(count_avc + np.random.laplace(0, 2 * (data.shape[1]- 1) / epsilon, 2) * dp)
                    #count_a_list.append(sum(count_avc))
                    Ha_v = entropy(count_avc / sum(count_avc))
                    paHa = paHa + Ha_v*pa[i]
                HSA.append(paHa)
            IG_alist = HS-np.array(HSA)

            # Find the attribute with highest IG
            max_ind = np.argmax(IG_alist)
            real_ind = ind_list[max_ind]
            # print("found the most informative attribute index = ", att_ind)
            att_values = np.unique(data[:, max_ind])
            data_groups = split_data(data, max_ind, att_values)
            sublist = ind_list[0:max_ind] + ind_list[max_ind + 1:]
            for i in range(len(att_values)):
                self.create_child(real_ind, att_values[i], data_groups[i], sublist, dp, epsilon)
            # print("Child finished")
        else:
            num_att = data.shape[1]
            # There is only one attribute, only one record, only one class or reach maximum depth, create leaves.
            att_values = np.unique(data[:, 0])
            for i in range(len(att_values)):
                classes = data[data[:, 0] == att_values[i], -1]
                noise = np.random.laplace(0, 1/epsilon, 2)*dp
                counts = abs(np.array([sum(classes == 0), sum(classes == 1)]) + noise)
                prob = counts/sum(counts)
                self.create_leaf(ind_list[0], att_values[i], prob, noise)

    def predict_proba(self, X):
        result = []
        for x in X:
            result.append(self.prob_test(x))
        return np.array(result)

    def predict(self, X):
        result = []
        for x in X:
            prob = self.prob_test(x)
            result.append(np.argmax(prob))
        return np.array(result)

    def fit(self, X, y, dp=0, epsilon=0.01):
        data = np.hstack([np.array(X), np.array([y]).T])
        ind_list = list(range(data.shape[1]))
        return self.build_tree_from_subdata(data, ind_list, dp, epsilon)

    def clf_prob(self, X):
        return self.predict_proba(X), self.predict(X)

    def print_tree(self):
        text = ""
        text = plot_current(text, self, level=1)
        return text

    def Sum_noise(self):
        noise = 0
        noise += self.noise.sum()
        if self.child is None:
            return noise
        for c in self.child:
            noise += c.Sum_noise()
        return noise

    def Search_node(self, ind, value):
        noise = 0
        if self.child is None:
            return 0
        for c in self.child:
            if c.attrindex == ind and c.attrvalue == value:
                noise += np.abs(c.noise).sum()
            elif c.attrindex == ind and c.attrvalue != value:
                noise += 0
            else:
                noise = noise + c.Search_node(ind, value)
        return noise


def plot_current(text, node, level):
    new_string = "|" + "   |" * (level - 1) + '---feature' + str(node.attrindex) + ' = ' + str(node.attrvalue) + "\n"
    text = text + new_string
    if node.prob is None:
        for c in node.child:
            text = plot_current(text, c, level+1)
    else:
        new_string = "|" + "   |"*(level) + "---prob=" + str(node.prob) + "\n"
        text = text + new_string
    return text



## define a function to split data according to one attribute
def split_data(data, attrindex, att_values):
    # Copy the data with out the most informative attribute
    data_withoutattr = np.delete(data, attrindex, axis=1)
    informative_attr = data[:, attrindex]
    # Innitialize data groups and add each group to the list
    data_groups = []
    for v in att_values:
        data_groups.append(data_withoutattr[informative_attr==v,:])
    return data_groups