import math
import pandas as pd


def class_count(one_list: list):    # 计算分类
    return {_vc: one_list.count(_vc) for _vc in set(one_list)}


def class_percent(count_dict: dict):
    _sum = sum(count_dict.values())
    return {_k: 100 * _v / _sum for _k, _v in count_dict.items()}


def calc_entropy(one_list: list):   # 计算熵
    _len = len(one_list)
    _count = class_count(one_list)
    entropy = 0
    for num in _count.values():
        p = num / _len
        entropy -= p * math.log(p, 2)
    return entropy


def split_dataset(dataset, axis, value):  # 返回按某个特征分类后的数据
    return pd.DataFrame([
        [_v for _i, _v in enumerate(row) if _i != axis]
        for row in dataset.values.tolist() if row[axis] != value
    ])


def choose_dataset(dataset, axis, value):    # 返回按某个特征选择后的数据
    return pd.DataFrame([
        [_v for _i, _v in enumerate(row) if _i != axis]
        for row in dataset.values.tolist() if row[axis] == value
    ])


def choose_best_feature(dataset, width, base_entropy):
    best_info_gain = 0  # 信息增益
    best_feat_index = -1  # 最优特征下标
    best_feat_values = []
    v_len = len(dataset)
    for i in range(width):
        unique_feat_values = set(dataset.iloc[:, i])  # 用集合去重第i列特征组成的列表，得到特征值，如{'短', '长'}
        new_entropy = 0
        for value in unique_feat_values:  # 用特征值中的每一个 划分数据集
            sub_data_set = split_dataset(dataset, i, value)
            if not sub_data_set.empty:
                weight = len(sub_data_set) / v_len  # 权重，子集个数/ 全集个数
                new_entropy += weight * calc_entropy(
                    list(sub_data_set.iloc[:, width - 1]))  # 按某个特征分类后的熵 = 累加子集熵*weight
        if (infoGain := (base_entropy - new_entropy)) > best_info_gain:
            best_info_gain = infoGain
            best_feat_index = i
            best_feat_values = unique_feat_values
    return best_feat_index, best_feat_values


def majorityCnt(OneColumnDataSet: list):  # 按分类后类别数量排序
    return sorted(class_count(OneColumnDataSet).items(), key=lambda x: x[1], reverse=True)[0][0]


class TreeNode:
    # 无序决策树子节点
    def __init__(self, feat=None, value=None, label=None, children: dict = None):
        self.next_label = None
        self.label = label  # 分类依据
        self.feat = feat  # 分类值
        self.value = value  # 分类后的数据类分布
        self.children = children  # 孩子节点, 以字典储存

    @property
    def children(self):
        return self.__children

    @children.setter
    def children(self, value):
        self.__children = value if isinstance(value, dict) else {}

    def add_child(self, node: "TreeNode" = None):
        assert (c_feat := node.feat) not in self.children, "对应决策标签已存在"
        if self.next_label is None:
            self.next_label = node.label
        elif self.next_label != node.label:
            raise TypeError("标签不一致,不能用于决策树")
        self.children[c_feat] = node

    def goto(self, feat):
        assert feat in self.children, "不存在对应决策"
        return self.children[feat]

    def __str__(self):
        if self.value is None:
            return f"{self.feat}"
        if isinstance(self.value, dict):
            return f"{self.feat}{list(self.value.items())}"
        return f"{self.feat}({self.value})"

    def __repr__(self):
        return self.__str__()


class Tree:
    def __init__(self, root: TreeNode = None):
        self.root = self.index = root

    def add_child(self, node: TreeNode = None):
        if self.root is None:
            self.root = self.index = node
        else:
            self.index.add_child(node)

    def goto(self, label):
        self.index = self.index.goto(label)
        return self.index

    def show(self):
        print(self.root)
        temp_list = [self.root]
        while temp_list:
            t_temp_list = []
            for temp_index in temp_list:
                if isinstance(temp_index, TreeNode) and temp_index.children:
                    print('{', ' | '.join(map(str, temp_index.children.values())), '}', sep='', end=' ')
                    t_temp_list.extend(temp_index.children.values())
            temp_list = t_temp_list
            print()


class DT(Tree):
    def __init__(self):
        super().__init__()
        self.base_entropy = 0.0
        self.v_len = 0
        self.label = None
        self.dataset = None

    @property
    def width(self):
        return len(self.label)

    @property
    def X(self):
        return self.dataset.iloc[:, 0:self.width]

    @property
    def Y(self):
        return self.dataset.iloc[:, -1]

    def fit(self, X, Y=None, label: list = None):
        # X, Y必须为二维列表且长度相等
        if Y is None:
            self.dataset = pd.DataFrame(X)
        else:
            dataset = [row_x + row_y for row_x, row_y in zip(X, Y)]
            self.dataset = pd.DataFrame(dataset)
        self.label = label
        self.v_len = len(self.dataset)
        self.base_entropy = calc_entropy(list(self.Y))  # 原始的信息熵
        self.root = self.index = TreeNode(feat="ALL", label="ALL", value=class_count(list(self.Y)))
        self.__create_tree(self.dataset, self.label, self.width)
        ...

    def predict(self, x: list, label: list = None):
        label = label or self.label
        self.index = self.root
        check_true = 1
        while check_true:
            check_true = 0
            if self.index.next_label in label:
                check_true += 1
                self.goto(x[label.index(self.index.next_label)])
            if self.index.next_label == "Value":
                check_true += 1
                self.goto("Value")
            if self.index.next_label is None:
                return class_percent(self.index.value)
        raise TypeError("未知标签, 未找到对应的路径")

    def __create_tree(self, dataset, label, width):
        # print(dataset)
        # label = label.copy()
        new_y = list(dataset.iloc[:, -1])
        new_y_set = set(new_y)
        if len(new_y_set) == 1 or width == 0:  # 只剩一个类或无法划分
            return self.add_child(TreeNode(feat='Value', label='Value', value=self.index.value))
        # if width == 0:  # 没有标签划分
        #     return self.add_child(TreeNode(feat='Value', label='Value', value=self.index.value))
        best_feat_index, best_feat_values = choose_best_feature(dataset, width, self.base_entropy)
        best_feat = label[best_feat_index]
        n_label = [_v for _v in label if _v != best_feat]
        index_temp = self.index
        for best_feat_value in best_feat_values:
            self.index = index_temp
            _dataset = choose_dataset(dataset, best_feat_index, best_feat_value)
            self.add_child(
                TreeNode(
                    feat=best_feat_value, label=best_feat, value=class_count(list(_dataset.iloc[:, -1]))
                ))
            self.index = index_temp.children[best_feat_value]
            self.__create_tree(_dataset, n_label, width - 1)


if __name__ == '__main__':
    t = DT()
    t.fit([
        ["长", "粗", "硬", True],
        ["长", "细", "硬", False],
        ["短", "细", "硬", True],
        ["长", "粗", "软", False],
        ["长", "粗", "软", True],
        ["长", "粗", "软", True],
    ], label=["长短", "粗细", "软硬"])
    t.show()
    r = t.predict(["长", "粗", "软"])
    print(r)
