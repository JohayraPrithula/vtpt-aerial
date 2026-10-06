"""
Canonical class-name mapping for cross-dataset aerial domain adaptation.

Each of the four datasets (AID, CLRS, NWPU, UCM) names the same semantic
class differently on disk (e.g. "StorageTanks" / "storage tank" /
"storagetanks").  This module provides:

  * AERIAL_DATASET_DIRS : domain -> sub-directory that holds class folders
  * FOLDER_TO_CANONICAL : domain -> {on_disk_folder_name: canonical_name}
  * helpers to derive each domain's canonical class set and the closed-set
    intersection across an arbitrary list of domains.

Confirmed synonym decisions:
  * port  <-  harbor            (NWPU "harbor", UCM "harbor")
  * farmland <- agricultural    (UCM "agricultural")
"""

# Where the class folders actually live under cfg.DATASET.ROOT.
# NOTE: AID points at AID_dataset/data so the mirrored
#       AID_dataset/.cache/huggingface/... tree is never crawled.
AERIAL_DATASET_DIRS = {
    "AID":  "AID_dataset/data",
    "CLRS": "CLRS_dataset",
    "NWPU": "NWPU_dataset",
    "UCM":  "UCM_dataset",
}

FOLDER_TO_CANONICAL = {
    "AID": {
        "Airport": "airport",
        "BareLand": "bare_land",
        "BaseballField": "baseball_field",
        "Beach": "beach",
        "Bridge": "bridge",
        "Center": "center",
        "Church": "church",
        "Commercial": "commercial",
        "DenseResidential": "dense_residential",
        "Desert": "desert",
        "Farmland": "farmland",
        "Forest": "forest",
        "Industrial": "industrial",
        "Meadow": "meadow",
        "MediumResidential": "medium_residential",
        "Mountain": "mountain",
        "Park": "park",
        "Parking": "parking",
        "Playground": "playground",
        "Pond": "pond",
        "Port": "port",
        "RailwayStation": "railway_station",
        "Resort": "resort",
        "River": "river",
        "School": "school",
        "SparseResidential": "sparse_residential",
        "Square": "square",
        "Stadium": "stadium",
        "StorageTanks": "storage_tank",
        "Viaduct": "viaduct",
    },
    "CLRS": {
        "airport": "airport",
        "bare land": "bare_land",
        "beach": "beach",
        "bridge": "bridge",
        "commercial": "commercial",
        "desert": "desert",
        "farmland": "farmland",
        "forest": "forest",
        "golf course": "golf_course",
        "highway": "highway",
        "industrial": "industrial",
        "meadow": "meadow",
        "mountain": "mountain",
        "overpass": "overpass",
        "park": "park",
        "parking": "parking",
        "playground": "playground",
        "port": "port",
        "railway": "railway",
        "railway station": "railway_station",
        "residential": "residential",
        "river": "river",
        "runway": "runway",
        "stadium": "stadium",
        "storage tank": "storage_tank",
    },
    "NWPU": {
        "airplane": "airplane",
        "airport": "airport",
        "baseball diamond": "baseball_field",
        "basketball court": "basketball_court",
        "beach": "beach",
        "bridge": "bridge",
        "chaparral": "chaparral",
        "church": "church",
        "circular farmland": "circular_farmland",
        "cloud": "cloud",
        "commercial area": "commercial",
        "dense residential": "dense_residential",
        "desert": "desert",
        "forest": "forest",
        "freeway": "freeway",
        "golf course": "golf_course",
        "ground track field": "ground_track_field",
        "harbor": "port",
        "industrial area": "industrial",
        "intersection": "intersection",
        "island": "island",
        "lake": "lake",
        "meadow": "meadow",
        "medium residential": "medium_residential",
        "mobile home park": "mobile_home_park",
        "mountain": "mountain",
        "overpass": "overpass",
        "palace": "palace",
        "parking lot": "parking",
        "railway": "railway",
        "railway station": "railway_station",
        "rectangular farmland": "rectangular_farmland",
        "river": "river",
        "roundabout": "roundabout",
        "runway": "runway",
        "sea ice": "sea_ice",
        "ship": "ship",
        "snowberg": "snowberg",
        "sparse residential": "sparse_residential",
        "stadium": "stadium",
        "storage tank": "storage_tank",
        "tennis court": "tennis_court",
        "terrace": "terrace",
        "thermal power station": "thermal_power_station",
        "wetland": "wetland",
    },
    "UCM": {
        "agricultural": "farmland",
        "airplane": "airplane",
        "baseballdiamond": "baseball_field",
        "beach": "beach",
        "buildings": "buildings",
        "chaparral": "chaparral",
        "denseresidential": "dense_residential",
        "forest": "forest",
        "freeway": "freeway",
        "golfcourse": "golf_course",
        "harbor": "port",
        "intersection": "intersection",
        "mediumresidential": "medium_residential",
        "mobilehomepark": "mobile_home_park",
        "overpass": "overpass",
        "parkinglot": "parking",
        "river": "river",
        "runway": "runway",
        "sparseresidential": "sparse_residential",
        "storagetanks": "storage_tank",
        "tenniscourt": "tennis_court",
    },
}


def get_canonical_classes(domain):
    """Return the set of canonical class names available in a domain."""
    if domain not in FOLDER_TO_CANONICAL:
        raise ValueError(f"Unknown aerial domain: {domain}")
    return set(FOLDER_TO_CANONICAL[domain].values())


def compute_shared_classes(domains):
    """Closed-set intersection of canonical classes across `domains`.

    Returns a sorted list, which fixes a deterministic, contiguous
    0..C-1 label ordering shared by every domain.
    """
    sets = [get_canonical_classes(d) for d in domains]
    if not sets:
        return []
    shared = set.intersection(*sets)
    return sorted(shared)
