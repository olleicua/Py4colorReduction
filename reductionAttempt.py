class Node :
	"""
	Use:
	>>> Node("a")
	a_5
	>>> Node("b", 7)
	b_7
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
		self.color = None

	def __repr__(self) :
		return "%s_%d" % (self.name, self.degree)

	def __cmp__(self, other) :
		return cmp(self.name, other.name)

class BoundaryNode :
	"""
	A node with unspecified degree for the boundary ring of a configuration.
	Use:
	->>> BoundaryNode()
	b0
	->>> BoundaryNode()
	b1
	"""
	count = 0
	def __init__(self) :
		self.name = "b%d|" % self.__class__.count
		self.__class__.count += 1
		self.color = None

	def __repr__(self) :
		return self.name
	
class Configuration :
	"""
	The initial configurations should be already triangulated as we have no
	 clearly defined algorithm to triangulate them.
	"""
	def __init__(self, nodes, adjacencyList) :
		"""
		nodes is a list of Node objects
		adjacencyList is a list of pairs (sorted tuples) of nodes names
		Use:
		>>> c = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                   [("a", "b"), ("a", "c")])
		>>> c.nodes["a"]
		a_5
		>>> c.adjacencyList
		[(a_5, b_7), (a_5, c_8)]
		"""
		# create a dictionary mapping node names to nodes
		self.nodes = {}
		for node in nodes :
			self.nodes[node.name] = node

		# build adjacency list using nodes instead of node names
		self.adjacencyList = []
		for nodeA, nodeB in adjacencyList :
			self.adjacencyList.append(tuple(sorted((self.nodes[nodeA], self.nodes[nodeB]))))
	
	def __getitem__(self, name) :
		return self.nodes[name]
	
	def addEdge(self, nodeA, nodeB) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7)], [])
		>>> config.addEdge(config["a"], config["b"])
		>>> config.adjacencyList
		[(a_5, b_7)]
		>>> config.addEdge(config["a"], config["b"])
		>>> config.adjacencyList
		[(a_5, b_7)]
		"""
		if not tuple(sorted((nodeA, nodeB))) in self.adjacencyList :
			self.adjacencyList.append(tuple(sorted((nodeA, nodeB))))
		
	def getNeighbors(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                        [("a", "b"), ("a", "c")])
		>>> config.getNeighbors(config["a"])
		[b_7, c_8]
		"""
		result = []
		for nodeA, nodeB in self.adjacencyList :
			if node == nodeA :
				result.append(nodeB)
			if node == nodeB :
				result.append(nodeA)
		return result

	def apparentDegree(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                        [("a", "b"), ("a", "c")])
		>>> config.apparentDegree(config["a"])
		2
		"""
		return len(self.getNeighbors(node))
	
	def isOuter(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d"),
		...                         Node("e"), Node("f")],
		...                        [("a", "b"), ("a", "c"), ("a", "d"),
		...                         ("a", "e"), ("a", "f")])
		>>> config.isOuter(config["a"])
		False
		>>> config.isOuter(config["b"])
		True
		"""
		if isinstance(node, BoundaryNode) :
			return True
		return node.degree > len(self.getNeighbors(node))
	
	def outerNodes(self) :
		"""
		Return a list of nodes that have currently unspecified edges.
		Use:
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d"),
		...                         Node("e"), Node("f")],
		...                        [("a", "b"), ("a", "c"), ("a", "d"),
		...                         ("a", "e"), ("a", "f")])
		>>> sorted(map(lambda n: n.name, config.outerNodes()))
		['b', 'c', 'd', 'e', 'f']
		"""
		result = []
		for node in sorted(self.nodes.values()) :
			if self.isOuter(node) :
				result.append(node)
		return result
	
	def addBoundary(self) :
		"""
		Add boundary nodes (assumes that the configuartion is triangulated).
		 In cases like:
		     (5) -- (7) -- (5)
		 the behavior of this function will be undefined.
		Use:
		>>> config = Configuration([Node("a")], [])
		>>> config.addBoundary()
		>>> sorted(config.adjacencyList)
		[(a_5, b2|), (a_5, b1|), (a_5, b0|), (a_5, b3|), (a_5, b4|), (b2|, b1|), (b2|, b3|), (b1|, b0|), (b3|, b4|), (b4|, b0|)]
		"""
		# TODO : find a way to get the "outer neighbors of a node"
		outerNodes = self.outerNodes()
		for node in outerNodes :
			newCount = node.degree - self.apparentDegree(node)
			newBoundaryNodes = [BoundaryNode() for n in range(newCount)]
			for i, boundaryNode in enumerate(newBoundaryNodes) :
				self.nodes[boundaryNode.name] = boundaryNode
				self.addEdge(boundaryNode, node)
				if i > 0 :
					self.addEdge(boundaryNode, newBoundaryNodes[i - 1])
			outerNeighbors = [node for n in self.getNeighbors(node) \
							  if self.isOuter(n)]
			#print self.getNeighbors(node)
			self.addEdge(outerNeighbors[0], newBoundaryNodes[0])
			if outerNeighbors[0] != outerNeighbors[1] :
				self.addEdge(outerNeighbors[1], newBoundaryNodes[-1])

if __name__ == "__main__" :
	import doctest
	doctest.testmod()
