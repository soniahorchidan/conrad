from abc import ABC, abstractmethod


class AbstractDBController(ABC):
    """
    Controller for interacting with the database. This class defines the methods that are required for interacting with the database.
    """

    def __init__(self):
        """
        Initializes the database controller with the provided database path.

        Args:
            db_path (str): The path to the database.
        """
        self._prefix = "_orbDB_"

    @abstractmethod
    def connect_to_db(self, db_path):
        """
        Connects to the database and returns a driver object.

        Args:
            uri (str): The URI of the database.

        Returns:
            An object representing the connection to the database.
        """
        raise NotImplementedError

    @abstractmethod
    def clear_cache(self):
        """
        Clears the cache of the controller.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_node_ids(self):
        """
        Retrieves all nodes from the graph. Cache mechanism can be used to store the node IDs.

        Returns:
            list: A list containing the IDs of all nodes in the graph.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_relations(self):
        """
        Retrieves all relations from the graph.

        Returns:
            list: A list containing the IDs of all relations in the graph.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_num_relations(self):
        """
        Retrieves the number of relations in the graph.

        Returns:
            int: The number of relations in the graph.
        """
        raise NotImplementedError

    @abstractmethod
    def get_node_attributes(self, nids: list):
        """
        Retrieves the attributes of nodes with the specified IDs from the graph.

        Parameters:
        - nids (list): A list of node IDs.

        Returns:
        - dict: A dictionary representing the retrieved node from the graph. If no node is found
                with the specified ID, it returns an empty dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def get_whole_graph(self):
        """
        Retrieves the whole graph from the database.

        Returns:
        - list: A list containing the triples of the graph.
        """
        raise NotImplementedError

    @abstractmethod
    def set_property(self, nids: list, new_prop: str, prop_value: str):
        """
        Sets a new property with a given value for a node in the graph.

        Args:
            nids (list): A list of node IDs.
            new_prop (str): The name of the new property to set.
            prop_value (str): The value to assign to the new property.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def remove_property(self, nids: list, property: str):
        """
        Removes the specified property from the backend db nodes for the given node IDs.

        Args:
            nids (list): A list of node IDs. If the list is None, the property is removed from all nodes.
            property (str): The property to be removed from the backend db nodes.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def is_anchor_node(self, nid):
        """
        Checks if a node with the specified ID is an anchor node in the graph.

        Parameters:
        - nid (int): The ID of the node to check.

        Returns:
        - bool: True if the node with the specified ID is an anchor node, False otherwise.
                If the node with the specified ID is not found in the graph, it returns False.
        """
        raise NotImplementedError

    @abstractmethod
    def get_anchor_node_ids(self):
        """
        Retrieves all anchor nodes from the graph.

        Returns:
        - list: A list containing the IDs of all anchor nodes in the graph.
        """
        raise NotImplementedError

    @abstractmethod
    def write_isAnchor_property(self, ids: list, value: str):
        """
        Writes the 'isAnchor' property to nodes in the graph based on a list of node IDs.

        Args:
            ids (list): A list of node IDs.
            value (str): The value to be written in the backend db.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def get_random_node_ids(self, n: int):
        """
        Get a list of random node IDs.

        Args:
            n (int): The number of samples to return.

        Returns:
            list: A list of random node IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_one_hop_neighbours(
        self, nid, k=None, flow="both", ret_rels=False, merge_and_uniq=False
    ):
        """
        Retrieves up to k one-hop neighbors of a given node from the database.

        Args:
            nid (int or list): The ID of the node whose one-hop neighbors are to be retrieved.
            k (int): The maximum number of one-hop neighbors to retrieve. If k is None, all one-hop neighbors are returned.
            flow (str): The direction of the edge flow. Possible values are "both", "incoming", and "outgoing".
            ret_rels (bool): Whether to return the edge types of the one-hop neighbors.
            merge_and_uniq (bool): Whether to merge and return unique one-hop neighbors.

        Returns:
            list: A list of IDs tuples representing the edges between the node N and its one-hop neighbours. If ret_rels is True, the edge types are also returned in the form of a tuple.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_two_hop_neighbours(self, nid, one_hop_ns, maxx):
        """
        Retrieves up to max2 of the second-degree neighbors of a given node from the database.

        Args:
            nid (int): The ID of the node whose second-degree neighbors are to be retrieved.
            one_hop_ns (list): The IDs of the one-hop neighbors of the node.
            maxx (int): The maximum number of one-degree neighbors to retrieve for each one-hop neighbour.

        Returns:
            list: A list of IDs tuples representing the edges between the node N and its second-degree neighbors.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_neighbourhood(self, nid, pct):
        """
        Retrieves both one-hop and second-degree neighbors of a given node from the Neo4J database.

        Args:
            nids (list): The IDs of the nodes whose neighbors are to be retrieved.
            maxx (int): The maximum number of neighbors to retrieve at each hop.

        Returns:
            list: A list containing the one-hop neighbors and two-hop neighbors for each node in nids.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_dists_to_anchors_property(self, nids: list, node_attr: dict = None):
        """
        Retrieves the distances to anchor nodes property of nodes with the specified IDs from the graph.

        Parameters:
        - nids (list): A list of node IDs.
        - node_attr (dict): A dictionary containing the attributes of the nodes.

        Returns:
        - dict: A dictionary representing the distances to anchor nodes property of the node from the graph. If no node is found
                with the specified ID, it returns None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_relation_context(self, nids: list, node_attr: dict = None):
        """
        Retrieves the relation context of nodes with the specified IDs from the graph.

        Parameters:
        - nids (list): A list of node IDs.
        - node_attr (dict): A dictionary containing the attributes of the nodes.

        Returns:
        - dict: A dictionary representing the relation context of the node from the graph. If no node is found
                with the specified ID, it returns an empty dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def get_neighbours_edge_types(self, nid, k=None, flow="both", unique=True):
        """
        Retrieves edge types of neighboring nodes connected to a given node.

        Args:
            nid (int): The ID of the node.
            k (int): The maximum number of returned edge types. If k is None, all edge types are returned.
            flow (str): The direction of the edge flow. Possible values are "both", "incoming", and "outgoing".
            unique (bool): Whether to return unique edge types.

        Returns:
            list: A list containing unique edge types (up to r) of neighboring nodes connected to the given node.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_dists_to_anchors_property(self, dists: dict):
        """
        Writes the 'distsToAnchors' property to nodes in the graph based on a list of node IDs.

        Args:
            ids (list): A list of node IDs.
            dists (dict): The distances to be written in the backend db.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_distance_to_anchor_nodes(self, nid, anchors):
        """
        Calculates the distances between the specified node and all anchor nodes in the graph.

        Parameters:
        - nid (int): The ID of the node for which distances are calculated.

        Returns:
        - dict: A dictionary where keys are the IDs of anchor nodes and values are the distances
                between the specified node and the anchor nodes. If the specified node is an
                anchor node, its distance to itself is 0.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_relation_context(self, relation_context):
        """
        Writes the relation context to the database.

        This method iterates over the keys in the given `relation_context` dictionary.
        For each node identifier (`nid`), if the associated list of relations is not empty,
        it sets the property "relationContext" of the node with `nid` to the corresponding list of relations.

        Args:
            relation_context (dict): A dictionary where keys are node identifiers (nids)
                                     and values are lists of relation contexts associated with each node.

        Returns:
            None
        """
        raise NotImplementedError()

    def multi_hop_query_retrieval(self, nid: int, rids: list):
        """
        Retrieves a list of unique entity IDs by performing a multi-hop query on a Neo4j database.
        This function executes a query that matches a pattern of entities connected by specified
        relationships starting from an entity with a given ID (`nid`). The query traverses three
        relationships (specified by `rids`) and retrieves the distinct IDs of the final entities.
        Parameters:
        - nid (int) : The ID of the starting entity.
        - rids (list) : A list of n relationship types to traverse. Each element in the list corresponds
                        to the type of relationship for each hop in the query.
        Returns:
            list:  A list of distinct IDs of the final entities reached after three hops.
        """

    @abstractmethod
    def sample_queries(self, hops: int, batch_size: int, negative_sample_size: int):
        """
        Sample queries from the database.

        Args:
            hops (int): The number of hops in the query.
            batch_size (int): The number of queries to sample.
            negative_sample_size (int): The number of negative samples to generate for each query.

        Returns:
            queries (list): A list of queries. Each query is a list of node IDs and relationship IDs.
            tails (list): A list of node IDs. Each node ID is the true tail node for each query.
            negative_tails (list): A list of lists of node IDs. Each list contains the negative tail node IDs for each query.
            subsampling_weight (list): A list of weights for each query.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_queries_neg_relation(self, batch_size: int, negative_sample_size: int):
        """
        Sample queries from the database.

        Args:
            batch_size (int): The number of queries to sample.
            negative_sample_size (int): The number of negative samples to generate for each query.

        Returns:
            queries (list): A list of queries. Each query is a list of node IDs and relationship IDs.
            tails (list): A list of node IDs. Each node ID is the true tail node for each query.
            negative_relations (list): A list of lists of relationship IDs. Each list contains the negative relationship IDs for each query.
            subsampling_weight (list): A list of weights for each query.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_sub_graph(self, num_entities: int):
        """
        Sample a subgraph from the database.

        Args:
            num_entities (int): The number of entities to sample.

        Returns:
            triples (list): A list of triples. Each triple is a tuple of node IDs and relationship IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Closes the connection to the database.

        Returns:
            None
        """
        raise NotImplementedError()
