import numpy as np
from typing import List, Tuple, Sequence, Optional, Any, Dict

from neo4j import GraphDatabase, Driver
from .base import TSPSolverBase


class Neo4jSolver(TSPSolverBase):
    """
    Solveur TSP Path basé sur Neo4j.

    - Utilise TSPSolverBase pour la matrice OSRM et le calcul du coût.
    - Délègue la construction de la route à Neo4j via une requête Cypher.
    - Suppose l'existence d'un mapping index_matrice -> node_id Neo4j.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Matrice de coûts (OSRM), carrée.
    start : int
        Index du POI de départ (dans la matrice).
    uri : str
        URI Neo4j, ex: "bolt://localhost:7687".
    user : str
        Utilisateur Neo4j.
    password : str
        Mot de passe Neo4j.
    database : str, optional
        Nom de la base (Neo4j >= 4.x). None pour la base par défaut.
    index_to_neo4j_id : Sequence[Any]
        Liste (ou séquence) telle que index_to_neo4j_id[i] = node_id Neo4j du POI i.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        start: int,
        uri: str,
        user: str,
        password: str,
        database: Optional[str],
        index_to_neo4j_id: Sequence[Any],
        solver_name: str = "Neo4j",
    ):
        super().__init__(distance_matrix, start=start, name=solver_name)

        if len(index_to_neo4j_id) != self.n:
            raise ValueError(
                f"index_to_neo4j_id doit avoir la même taille que la matrice "
                f"({len(index_to_neo4j_id)} vs {self.n})"
            )

        self.index_to_neo4j_id = list(index_to_neo4j_id)
        self.database = database

        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    # ---------------------------------------------------------
    # Mapping indices <-> Neo4j node ids
    # ---------------------------------------------------------
    def _index_to_id(self, idx: int) -> Any:
        return self.index_to_neo4j_id[idx]

    def _id_to_index(self, node_id: Any) -> int:
        # On suppose que les node_ids sont uniques et présents dans la liste
        try:
            return self.index_to_neo4j_id.index(node_id)
        except ValueError:
            raise ValueError(f"node_id inconnu dans index_to_neo4j_id: {node_id!r}")

    # ---------------------------------------------------------
    # Construction de la requête Cypher
    # ---------------------------------------------------------
    def _build_cypher_query(self) -> str:
        """
        Construit la requête Cypher qui doit renvoyer une route.

        La requête doit:
          - prendre en entrée:
              $poi_ids   : liste des node_ids Neo4j des POIs à visiter
              $start_id  : node_id du POI de départ
          - renvoyer:
              une seule ligne avec une colonne `route_ids`
              qui est une liste ordonnée de node_ids représentant le chemin.

        Exemple d'API cible:

            RETURN [n IN route | id(n)] AS route_ids

        NOTE: à adapter selon l'implémentation réelle de vos collègues.
        """
        # ------- EXEMPLE SIMPLIFIÉ (placeholder) -------
        #
        # Ici, on met un heuristique greedy tout simple en Cypher
        # basé sur des relations (:POI)-[:ROUTE {cost: ...}]->(:POI)
        # avec un property `cost` représentant la distance / durée.
        #
        # IMPORTANT: À adapter à TON schéma réel (labels, propriétés, relations).
        #
        query = """
        // Heuristique greedy TSP path très simple, à adapter
        MATCH (start:POI)
        WHERE id(start) = $start_id

        // Récupérer tous les POIs à visiter
        MATCH (p:POI)
        WHERE id(p) IN $poi_ids
        WITH start, collect(p) AS pois

        // Initialisation
        WITH start, pois,
             [start] AS path,
             [start] AS visited

        // Boucle itérative en utilisant APOC pour simuler une boucle
        CALL apoc.periodic.iterate(
          '
          WITH start, pois, path, visited
          UNWIND range(1, size(pois)-1) AS step
          WITH path, visited, pois
          WITH path, last(path) AS current, visited, pois
          MATCH (current)-[r:ROUTE]->(cand:POI)
          WHERE cand IN pois AND NOT cand IN visited
          WITH path, visited, cand, r
          ORDER BY r.cost ASC
          WITH path + cand AS newPath, visited + cand AS newVisited
          RETURN newPath AS path, newVisited AS visited
          ',
          '',
          {batchSize:1, parallel:false}
        ) YIELD batches, total

        WITH last(path) AS finalPath
        RETURN [n IN finalPath | id(n)] AS route_ids
        """
        return query

    # ---------------------------------------------------------
    # Appel à Neo4j
    # ---------------------------------------------------------
    def _call_neo4j_route(self) -> List[int]:
        """
        Appelle Neo4j pour obtenir une route (liste d'indices de matrice).

        Étapes:
          1. Construire la requête Cypher
          2. Envoyer poi_ids (node_ids Neo4j) + start_id
          3. Récupérer route_ids (liste d'ids Neo4j)
          4. Convertir en indices de matrice
        """
        poi_ids = [self._index_to_id(i) for i in range(self.n)]
        start_id = self._index_to_id(self.start)

        query = self._build_cypher_query()

        def _run_tx(tx, q, params: Dict[str, Any]):
            result = tx.run(q, **params)
            record = result.single()
            if record is None:
                raise RuntimeError("La requête Neo4j n'a renvoyé aucun résultat.")
            route_ids = record.get("route_ids")
            if route_ids is None:
                raise RuntimeError("La requête Neo4j doit renvoyer une colonne 'route_ids'.")
            return route_ids

        with self._driver.session(database=self.database) as session:
            route_ids = session.execute_read(
                _run_tx,
                query,
                {"poi_ids": poi_ids, "start_id": start_id},
            )

        # Conversion node_ids -> indices de matrice
        route_indices = [self._id_to_index(node_id) for node_id in route_ids]

        # Sanity check: début = self.start, tous les POIs présents
        if route_indices[0] != self.start:
            raise ValueError(
                f"La route Neo4j ne commence pas par le start attendu "
                f"({route_indices[0]} vs {self.start})"
            )
        if sorted(route_indices) != list(range(self.n)):
            raise ValueError(
                f"La route Neo4j ne contient pas exactement tous les indices 0..{self.n-1} : "
                f"{sorted(route_indices)}"
            )

        return route_indices

    # ---------------------------------------------------------
    # solve()
    # ---------------------------------------------------------
    def solve(self) -> Tuple[List[int], float]:
        """
        Récupère une route via Neo4j, puis calcule son coût via la matrice OSRM.
        """
        route = self._call_neo4j_route()
        cost = self.route_cost(route)
        return route, cost

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def close(self):
        if self._driver is not None:
            self._driver.close()