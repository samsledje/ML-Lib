# Tree Data Structures

class Node:
    def __init__(self, key, value = None):
        self.key = key
        self.value = value
        self.children = []
    
    def __lt__(self, other):
        return self.key < other.key

    def __eq__(self, other):
        return self.key == other.key

    def __gt__(self, other):
        return self.key > other.key

class BinaryNode(Node):
    def __init__(self, key, value = None):
        super().__init__(key, value)
        self.lchild = None
        self.rchild = None
        self.children = [self.lchild, self.rchild]

    def insert(self, new):
        if self < new:
            if self.lchild is not None:
                self.lchild.insert(new)
            else:
                self.lchild = new
        else:
            if self.rchild is not None:
                self.rchild.insert(new)
            else:
                self.rchild = new
    
    def search(self, key):
        if self == key

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(key, value = None):
        new = BinaryNode(key, value)
        if self.root is None:
            self.root = new
        else:
            self.root.search_insert(new)
    
    def remove(key):
        pass

    def search(key):
        search = 
