class Node :
	"""
	Use:
	>>> Node("a")
	a(5)
	>>> Node("b", 7)
	b(7)
	>>> Node("c", 0)
	Traceback (most recent call last):
	    ...
	ValueError: degree must be at least 5
	"""
	def __init__(self, name, degree=5) :
		self.name = name
		if degree < 5 :
			raise ValueError("degree must be at least 5")
		self.degree = degree

	def __repr__(self) :
		return "%s(%d)" % (self.name, self.degree)
		
class Configuration :
	def __init__(self, nodes, adjacencyList) :
		"""
		nodes is a list of Node objects
		adjacencyList is a list of pairs (tuples) of nodes names
		Use:
		>>> c = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                   [("a", "b"), ("a", "c")])
		>>> c.nodes["a"]
		a(5)
		>>> c.adjacencyList
		[(a(5), b(7)), (a(5), c(8))]
		"""
		# create a dictionary mapping node names to nodes
		self.nodes = {}
		for node in nodes :
			self.nodes[node.name] = node

		# build adjacency list using nodes instead of node names
		self.adjacencyList = []
		for nodeA, nodeB in adjacencyList :
			self.adjacencyList.append((self.nodes[nodeA], self.nodes[nodeB]))

	def __getitem__(self, name) :
		return self.nodes[name]

	def getNeighbors(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                        [("a", "b"), ("a", "c")])
		>>> config.getNeighbors(config["a"])
		[b(7), c(8)]
		"""
		result = []
		for nodeA, nodeB in self.adjacencyList :
			if node == nodeA :
				result.append(nodeB)
			if node == nodeB :
				result.append(nodeA)
		return result

	def isBoundary(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d"),
		...                         Node("e"), Node("f")],
		...                        [("a", "b"), ("a", "c"), ("a", "d"),
		...                         ("a", "e"), ("a", "f")])
		>>> config.isBoundary(config["a"])
		False
		>>> config.isBoundary(config["b"])
		True
		"""
		return node.degree > len(self.getNeighbors(node))
	
if __name__ == "__main__" :
	import doctest
	doctest.testmod()
