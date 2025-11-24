from sqlalchemy import Engine

from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter


def test_my_adapter_function(
    matchbox_postgres: MatchboxDBAdapter, sqlite_warehouse: Engine
) -> None:
    """Test scenario."""
    with setup_scenario(matchbox_postgres, "link", sqlite_warehouse) as dag:
        # The backend is now populated with the 'link' scenario
        # dag.sources contains the source testkits
        # dag.models contains the model testkits

        # Now you can call the function you want to test
        results = matchbox_postgres.clusters.count()

        print("\nGet examples of source and model resolutions")
        # s = dag.sources["crn"].resolution_path  # source resolution
        m = dag.sources["cdms"].resolution_path  # model resolution
        # print(f"Source: {s}")
        print(f"Model: {m}")
        clusters_data = matchbox_postgres.data.count()
        clusters_clusters = matchbox_postgres.clusters.count()
        print(f"data before: {clusters_data}")
        print(f"data before: {clusters_clusters}")

        print("\nDelete resolutions")
        # matchbox_postgres.delete_resolution(s, certain=True)
        matchbox_postgres.delete_resolution(m, certain=True)

        print("\nDelete orphans")
        a = matchbox_postgres.delete_orphans()
        print(a)
        clusters = matchbox_postgres.data.count()
        print(f"clusters after: {clusters}")

        # You can use the dag to verify the results
        assert True
