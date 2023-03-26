class Node:
    def __init__(self, value=None, attr=None):
        self.value = value
        self.attr = attr
        self.children = {}

    def add_child(self, attribute_val: int, new_child):
        self.children[attribute_val] = new_child

    def is_leaf(self):
        if not self.children:
            return True
        return False